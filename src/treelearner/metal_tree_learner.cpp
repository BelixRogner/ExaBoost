/*!
 * Copyright (c) 2026 ExaBoost authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "metal_tree_learner.h"

#ifdef USE_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/log.h>

#include "feature_histogram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace LightGBM {

namespace {

constexpr int kThreadsPerGroup = 256;
constexpr int kMaxNumBins      = 256;

// Inlined Metal kernel source. Kept in sync with src/treelearner/metal/histogram256.metal;
// a regression test verifies the two strings match.
const char* kHistogramKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

#ifndef NUM_BINS
#define NUM_BINS 256u
#endif
#ifndef NUM_SUBHIST
#define NUM_SUBHIST 4u
#endif

inline void atomic_tg_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float cur = as_type<float>(expected);
        desired = as_type<uint>(cur + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed));
}

kernel void histogram_partial(
    device const uchar*  features      [[ buffer(0) ]],
    device const float*  gradients     [[ buffer(1) ]],
    device const float*  hessians      [[ buffer(2) ]],
    device float*        partial_hist  [[ buffer(3) ]],
    constant uint&       num_data      [[ buffer(4) ]],
    constant uint&       wg_per_feat   [[ buffer(5) ]],
    uint3 tid3      [[ thread_position_in_threadgroup ]],
    uint3 gid       [[ threadgroup_position_in_grid ]],
    uint3 tg_sz3    [[ threads_per_threadgroup ]],
    uint  sg_idx    [[ simdgroup_index_in_threadgroup ]])
{
    uint tid   = tid3.x;
    uint tg_sz = tg_sz3.x;
    threadgroup atomic_uint local_grad[NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_uint local_hess[NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat_id    = gid.y;
    uint wg_in_feat = gid.x;
    uint per_wg     = (num_data + wg_per_feat - 1) / wg_per_feat;
    uint start      = wg_in_feat * per_wg;
    uint end        = min(start + per_wg, num_data);
    device const uchar* feat_col = features + (uint64_t)feat_id * num_data;
    uint sub = sg_idx % NUM_SUBHIST;
    for (uint i = start + tid; i < end; i += tg_sz) {
        uint bin = (uint)feat_col[i];
        atomic_tg_add_f(&local_grad[sub][bin], gradients[i]);
        atomic_tg_add_f(&local_hess[sub][bin], hessians[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out =
        partial_hist + ((uint64_t)feat_id * wg_per_feat + wg_in_feat) * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        float g = 0.0f, h = 0.0f;
        for (uint s = 0; s < NUM_SUBHIST; ++s) {
            g += as_type<float>(atomic_load_explicit(&local_grad[s][b], memory_order_relaxed));
            h += as_type<float>(atomic_load_explicit(&local_hess[s][b], memory_order_relaxed));
        }
        out[2 * b + 0] = g;
        out[2 * b + 1] = h;
    }
}

kernel void histogram_reduce(
    device const float*  partial_hist  [[ buffer(0) ]],
    device float*        out_hist      [[ buffer(1) ]],
    constant uint&       wg_per_feat   [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint gid [[ threadgroup_position_in_grid ]])
{
    uint feat_id = gid;
    uint bin     = tid;
    float g = 0.0f, h = 0.0f;
    for (uint w = 0; w < wg_per_feat; ++w) {
        device const float* p =
            partial_hist + ((uint64_t)feat_id * wg_per_feat + w) * NUM_BINS * 2;
        g += p[2 * bin + 0];
        h += p[2 * bin + 1];
    }
    device float* out = out_hist + (uint64_t)feat_id * NUM_BINS * 2;
    out[2 * bin + 0] = g;
    out[2 * bin + 1] = h;
}

// Indexed variant: only rows whose global index is in `indices[0..num_idx)`
// contribute. Used for deeper leaves where data_indices is non-null.
kernel void histogram_partial_indexed(
    device const uchar*  features      [[ buffer(0) ]],
    device const float*  gradients     [[ buffer(1) ]],   // unordered, indexed by global row id
    device const float*  hessians      [[ buffer(2) ]],
    device float*        partial_hist  [[ buffer(3) ]],
    constant uint&       num_data      [[ buffer(4) ]],   // total rows (column stride)
    constant uint&       wg_per_feat   [[ buffer(5) ]],
    device const uint*   indices       [[ buffer(6) ]],   // [num_idx]
    constant uint&       num_idx       [[ buffer(7) ]],
    uint3 tid3      [[ thread_position_in_threadgroup ]],
    uint3 gid       [[ threadgroup_position_in_grid ]],
    uint3 tg_sz3    [[ threads_per_threadgroup ]],
    uint  sg_idx    [[ simdgroup_index_in_threadgroup ]])
{
    uint tid   = tid3.x;
    uint tg_sz = tg_sz3.x;
    threadgroup atomic_uint local_grad[NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_uint local_hess[NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat_id    = gid.y;
    uint wg_in_feat = gid.x;
    uint per_wg     = (num_idx + wg_per_feat - 1) / wg_per_feat;
    uint start      = wg_in_feat * per_wg;
    uint end        = min(start + per_wg, num_idx);
    device const uchar* feat_col = features + (uint64_t)feat_id * num_data;
    uint sub = sg_idx % NUM_SUBHIST;
    for (uint j = start + tid; j < end; j += tg_sz) {
        uint row = indices[j];
        uint bin = (uint)feat_col[row];
        atomic_tg_add_f(&local_grad[sub][bin], gradients[row]);
        atomic_tg_add_f(&local_hess[sub][bin], hessians[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out =
        partial_hist + ((uint64_t)feat_id * wg_per_feat + wg_in_feat) * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        float g = 0.0f, h = 0.0f;
        for (uint s = 0; s < NUM_SUBHIST; ++s) {
            g += as_type<float>(atomic_load_explicit(&local_grad[s][b], memory_order_relaxed));
            h += as_type<float>(atomic_load_explicit(&local_hess[s][b], memory_order_relaxed));
        }
        out[2 * b + 0] = g;
        out[2 * b + 1] = h;
    }
}
)MSL";

}  // namespace

// Set via LIGHTGBM_METAL_TIMING=1 — accumulates per-call timings so users can
// see where the GPU path is spending time. Reported once at process exit.
struct MetalTimings {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> idx_copy_us{0};
  std::atomic<uint64_t> dispatch_us{0};   // commit + waitUntilCompleted (wall clock)
  std::atomic<uint64_t> gpu_us{0};        // MTL::CommandBuffer GPU-only time
  std::atomic<uint64_t> writeback_us{0};
  bool enabled = false;
  ~MetalTimings() {
    if (enabled && calls.load() > 0) {
      uint64_t n = calls.load();
      std::fprintf(stderr,
        "[metal-timing] calls=%llu  idx_copy=%.2fms  dispatch=%.2fms  "
        "gpu_only=%.2fms  writeback=%.2fms  (per-call avg: "
        "idx_copy=%.2fus dispatch=%.2fus gpu_only=%.2fus writeback=%.2fus)\n",
        (unsigned long long)n,
        idx_copy_us.load() / 1000.0, dispatch_us.load() / 1000.0,
        gpu_us.load() / 1000.0, writeback_us.load() / 1000.0,
        (double)idx_copy_us.load() / n, (double)dispatch_us.load() / n,
        (double)gpu_us.load() / n, (double)writeback_us.load() / n);
    }
  }
};
static MetalTimings g_timings;

struct MetalTreeLearner::MetalState {
  MTL::Device*               device  = nullptr;
  MTL::CommandQueue*         queue   = nullptr;

  // Per-binsize compiled artifacts (idx 0=16-bin, 1=64-bin, 2=256-bin).
  struct PerBinSize {
    MTL::Library*              library             = nullptr;
    MTL::ComputePipelineState* pso_partial         = nullptr;
    MTL::ComputePipelineState* pso_partial_indexed = nullptr;
    MTL::ComputePipelineState* pso_reduce          = nullptr;
  };
  PerBinSize bs16, bs64, bs256;

  PerBinSize* active = nullptr;   // chosen at BuildDenseFeatureBuffer based on max_num_bin
  int active_bins = 0;            // 16 / 64 / 256

  // Persistent device-resident buffers (rebuilt on training-data changes).
  MTL::Buffer* feat_buf     = nullptr;  // uchar [num_metal_features * num_data]
  MTL::Buffer* grad_buf     = nullptr;  // float [num_data]
  MTL::Buffer* hess_buf     = nullptr;  // float [num_data]
  MTL::Buffer* idx_buf      = nullptr;  // uint  [num_data]
  MTL::Buffer* partial_buf  = nullptr;  // float [num_features * wg_per_feat * 2*active_bins]
  MTL::Buffer* out_buf      = nullptr;  // float [num_features * 2*active_bins]

  int num_metal_features = 0;
  int num_data           = 0;
  int wg_per_feat        = 1;

  static void ReleaseBinsize(PerBinSize* bs) {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(bs->pso_partial); rel(bs->pso_partial_indexed); rel(bs->pso_reduce);
    rel(bs->library);
  }

  ~MetalState() {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(feat_buf); rel(grad_buf); rel(hess_buf); rel(idx_buf);
    rel(partial_buf); rel(out_buf);
    ReleaseBinsize(&bs16); ReleaseBinsize(&bs64); ReleaseBinsize(&bs256);
    rel(queue); rel(device);
  }
};

MetalTreeLearner::MetalTreeLearner(const Config* tree_config)
    : SerialTreeLearner(tree_config),
      state_(std::make_unique<MetalState>()) {}

MetalTreeLearner::~MetalTreeLearner() = default;

void MetalTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  InitMetal();
  if (metal_ready_) BuildDenseFeatureBuffer();
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                              bool is_constant_hessian,
                                              bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian,
                                            reset_multi_val_bin);
  // Force a rebuild of the dense feature buffer next time around.
  metal_ready_ = false;
  metal_feature_groups_.clear();
}

void MetalTreeLearner::InitMetal() {
  // Honor LIGHTGBM_METAL_TIMING once globally (per process).
  if (const char* env = std::getenv("LIGHTGBM_METAL_TIMING")) {
    if (env[0] == '1') g_timings.enabled = true;
  }

  state_->device = MTL::CreateSystemDefaultDevice();
  if (!state_->device) {
    Log::Warning("Metal: no system default device found. Falling back to CPU path.");
    return;
  }
  Log::Info("Metal device: %s", state_->device->name()->utf8String());

  // Compile the kernel three times — once each for 16/64/256-bin variants.
  auto compile_variant = [&](MetalState::PerBinSize* bs, int num_bins,
                             int num_subhist) -> bool {
    NS::Error* e = nullptr;
    auto src = NS::String::string(kHistogramKernelSrc, NS::UTF8StringEncoding);
    auto opts = MTL::CompileOptions::alloc()->init();
    // preprocessorMacros: NSDictionary<NSString*, NSObject>.
    auto k_nb = NS::String::string("NUM_BINS",    NS::UTF8StringEncoding);
    auto k_ns = NS::String::string("NUM_SUBHIST", NS::UTF8StringEncoding);
    auto v_nb = NS::Number::number(num_bins);
    auto v_ns = NS::Number::number(num_subhist);
    const NS::Object* keys[2]   = { k_nb, k_ns };
    const NS::Object* values[2] = { v_nb, v_ns };
    auto macros = NS::Dictionary::dictionary(values, keys, 2);
    opts->setPreprocessorMacros(macros);
    bs->library = state_->device->newLibrary(src, opts, &e);
    opts->release();
    if (!bs->library) {
      Log::Warning("Metal: MSL compile (NUM_BINS=%d) failed: %s", num_bins,
                   e ? e->localizedDescription()->utf8String() : "(unknown)");
      return false;
    }
    auto make_pso = [&](const char* name) -> MTL::ComputePipelineState* {
      auto nm = NS::String::string(name, NS::UTF8StringEncoding);
      MTL::Function* fn = bs->library->newFunction(nm);
      if (!fn) { Log::Warning("Metal: missing kernel %s in NUM_BINS=%d library",
                              name, num_bins); return nullptr; }
      NS::Error* err2 = nullptr;
      auto p = state_->device->newComputePipelineState(fn, &err2);
      fn->release();
      if (!p) Log::Warning("Metal: PSO creation failed (%s, NUM_BINS=%d): %s",
                           name, num_bins,
                           err2 ? err2->localizedDescription()->utf8String() : "(unknown)");
      return p;
    };
    bs->pso_partial         = make_pso("histogram_partial");
    bs->pso_partial_indexed = make_pso("histogram_partial_indexed");
    bs->pso_reduce          = make_pso("histogram_reduce");
    return bs->pso_partial && bs->pso_partial_indexed && bs->pso_reduce;
  };
  bool ok = compile_variant(&state_->bs16,  16,  16)
         && compile_variant(&state_->bs64,  64,  8)
         && compile_variant(&state_->bs256, 256, 4);
  if (!ok) {
    Log::Warning("Metal: kernel compile incomplete. Falling back to CPU path.");
    return;
  }

  state_->queue = state_->device->newCommandQueue();
  metal_ready_ = true;
  Log::Info("Metal: backend initialized (16/64/256-bin kernels compiled, queue ready).");
}

void MetalTreeLearner::TeardownMetal() {
  state_.reset();
  metal_ready_ = false;
}

bool MetalTreeLearner::BuildDenseFeatureBuffer() {
  // Phase 2.1: a dataset is "Metal-eligible" if every feature group is a
  // single non-multi-val feature with <= NUM_BINS bins. For these we
  // materialize a packed [num_features × num_data] uchar buffer once.
  const int num_groups   = train_data_->num_feature_groups();
  const int num_features = train_data_->num_features();

  // Heuristic: Metal dispatch overhead dominates on very narrow datasets where
  // the GPU is underfilled. Empirically (Apple M4 Pro, tools/metal_bench/
  // train_bench.py), 64-feature workloads regress ~1.2x while 128+ features
  // see 1.1-1.4x speedups, so we conservatively keep Metal off below 96.
  // Override with LIGHTGBM_METAL_MIN_FEATURES=N.
  int min_features = 96;
  if (const char* env = std::getenv("LIGHTGBM_METAL_MIN_FEATURES")) {
    int v = std::atoi(env);
    if (v > 0) min_features = v;
  }
  if (num_features < min_features) {
    Log::Info("Metal: skipping acceleration (num_features=%d < min=%d). "
              "Override with LIGHTGBM_METAL_MIN_FEATURES.",
              num_features, min_features);
    return false;
  }

  // Multi-feature groups (LightGBM packs narrow features together) are OK —
  // FeatureIterator(f) abstracts the packing and we materialize a flat
  // per-feature column buffer.
  for (int g = 0; g < num_groups; ++g) {
    if (train_data_->IsMultiGroup(g)) {
      Log::Info("Metal: skipping acceleration (multi-val feature group %d).", g);
      return false;
    }
  }
  int max_num_bin = 0;
  for (int f = 0; f < num_features; ++f) {
    int nb = train_data_->FeatureNumBin(f);
    if (nb > kMaxNumBins) {
      Log::Info("Metal: skipping acceleration (feature %d has %d bins, > %d).",
                f, nb, kMaxNumBins);
      return false;
    }
    if (nb > max_num_bin) max_num_bin = nb;
  }

  // Pick the smallest kernel variant that fits all features.
  if (max_num_bin <= 16) {
    state_->active = &state_->bs16;  state_->active_bins = 16;
  } else if (max_num_bin <= 64) {
    state_->active = &state_->bs64;  state_->active_bins = 64;
  } else {
    state_->active = &state_->bs256; state_->active_bins = 256;
  }

  const data_size_t num_data = train_data_->num_data();
  metal_feature_groups_.resize(num_features);
  per_feature_num_bin_.resize(num_features);
  per_feature_offset_.resize(num_features);
  for (int f = 0; f < num_features; ++f) {
    metal_feature_groups_[f] = f;
    per_feature_num_bin_[f]  = train_data_->FeatureNumBin(f);
    per_feature_offset_[f]   =
        (train_data_->FeatureBinMapper(f)->GetMostFreqBin() == 0) ? 1 : 0;
  }

  // Materialize features into a single uchar buffer.
  const size_t feat_bytes = (size_t)num_features * (size_t)num_data;
  state_->feat_buf = state_->device->newBuffer(feat_bytes, MTL::ResourceStorageModeShared);
  if (!state_->feat_buf) {
    Log::Warning("Metal: failed to allocate feature buffer (%zu bytes).", feat_bytes);
    return false;
  }
  uint8_t* feat_ptr = static_cast<uint8_t*>(state_->feat_buf->contents());
  for (int f = 0; f < num_features; ++f) {
    std::unique_ptr<BinIterator> it(train_data_->FeatureIterator(f));
    uint8_t* col = feat_ptr + (size_t)f * num_data;
    // Use Get() (not RawGet) so multi-feature groups give per-sub-feature bin
    // values rather than the packed group byte. Get returns values in
    // [offset, num_bin) for in-range rows, or most_freq_bin for out-of-range.
    for (data_size_t i = 0; i < num_data; ++i) {
      col[i] = static_cast<uint8_t>(it->Get(i));
    }
  }

  // Allocate gradient/hessian shared buffers (zero-copy on Apple silicon).
  state_->grad_buf = state_->device->newBuffer((size_t)num_data * sizeof(score_t),
                                               MTL::ResourceStorageModeShared);
  state_->hess_buf = state_->device->newBuffer((size_t)num_data * sizeof(score_t),
                                               MTL::ResourceStorageModeShared);

  // Auto-tune wg_per_feat to target ~512 threadgroups (good occupancy on a
  // 20-core M-series GPU without underfilling on narrow datasets).
  // Override via LIGHTGBM_METAL_WG_PER_FEAT=N for tuning experiments.
  state_->wg_per_feat = std::max(1, std::min(32, 512 / std::max(num_features, 1)));
  if (const char* env = std::getenv("LIGHTGBM_METAL_WG_PER_FEAT")) {
    int v = std::atoi(env);
    if (v > 0) state_->wg_per_feat = std::min(v, 64);
  }
  state_->num_metal_features = num_features;
  state_->num_data = num_data;

  state_->partial_buf = state_->device->newBuffer(
      (size_t)num_features * (size_t)state_->wg_per_feat * state_->active_bins * 2 * sizeof(float),
      MTL::ResourceStorageModePrivate);
  state_->out_buf = state_->device->newBuffer(
      (size_t)num_features * state_->active_bins * 2 * sizeof(float),
      MTL::ResourceStorageModeShared);
  // Pre-allocate the indices buffer to avoid per-call allocation. Sized for
  // the worst case (all rows in a leaf).
  state_->idx_buf = state_->device->newBuffer((size_t)num_data * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);

  Log::Info("Metal: %d-feature dense buffer materialized (%.1f MB). "
            "wg_per_feat=%d, NUM_BINS=%d.",
            num_features, feat_bytes / 1048576.0, state_->wg_per_feat,
            state_->active_bins);
  return true;
}

void MetalTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  // Gradients/hessians are set once per tree by the caller before BeforeTrain.
  // Copy them into the shared Metal buffers exactly once here so the per-leaf
  // histogram dispatch doesn't re-pay the memcpy.
  if (metal_ready_ && !metal_feature_groups_.empty()) {
    std::memcpy(state_->grad_buf->contents(), gradients_,
                (size_t)state_->num_data * sizeof(score_t));
    std::memcpy(state_->hess_buf->contents(), hessians_,
                (size_t)state_->num_data * sizeof(score_t));
  }
}

void MetalTreeLearner::RunMetalHistogram(const score_t* /*gradients*/,
                                         const score_t* /*hessians*/,
                                         data_size_t num_data) {
  // grad/hess are already in shared buffers (copied once in BeforeTrain).

  uint32_t nd = (uint32_t)num_data;
  uint32_t wg = (uint32_t)state_->wg_per_feat;

  // Fast path: when wg_per_feat==1, the partial kernel already produces the
  // final histogram (no reduction needed). Skip the reduce dispatch.
  const bool one_wg = (state_->wg_per_feat == 1);
  MTL::Buffer* partial_dst = one_wg ? state_->out_buf : state_->partial_buf;

  MTL::CommandBuffer* cb = state_->queue->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(state_->active->pso_partial);
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->grad_buf, 0, 1);
  enc->setBuffer(state_->hess_buf, 0, 2);
  enc->setBuffer(partial_dst, 0, 3);
  enc->setBytes(&nd, sizeof(uint32_t), 4);
  enc->setBytes(&wg, sizeof(uint32_t), 5);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, state_->num_metal_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));

  if (!one_wg) {
    enc->setComputePipelineState(state_->active->pso_reduce);
    enc->setBuffer(state_->partial_buf, 0, 0);
    enc->setBuffer(state_->out_buf, 0, 1);
    enc->setBytes(&wg, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(
        MTL::Size::Make(state_->num_metal_features, 1, 1),
        MTL::Size::Make(state_->active_bins, 1, 1));
  }
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (g_timings.enabled) {
    double gpu_s = cb->GPUEndTime() - cb->GPUStartTime();
    if (gpu_s > 0) g_timings.gpu_us.fetch_add((uint64_t)(gpu_s * 1e6));
  }
}

void MetalTreeLearner::RunMetalHistogramIndexed(const data_size_t* data_indices,
                                                data_size_t num_idx) {
  // grad/hess already in shared buffers (BeforeTrain). Only the indices vary.
  // data_size_t is a 32-bit signed int storing non-negative row indices, which
  // is bitwise identical to uint32_t — memcpy is safe.
  static_assert(sizeof(data_size_t) == sizeof(uint32_t),
                "Metal indexed kernel assumes data_size_t is 32-bit.");
  auto t_idx0 = g_timings.enabled ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point();
  std::memcpy(state_->idx_buf->contents(), data_indices,
              (size_t)num_idx * sizeof(uint32_t));
  if (g_timings.enabled) {
    auto t1 = std::chrono::steady_clock::now();
    g_timings.idx_copy_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_idx0).count());
  }

  uint32_t nd  = (uint32_t)state_->num_data;
  uint32_t ni  = (uint32_t)num_idx;
  uint32_t wg  = (uint32_t)state_->wg_per_feat;

  // Same wg_per_feat==1 fast path as RunMetalHistogram (skip reduce).
  const bool one_wg = (state_->wg_per_feat == 1);
  MTL::Buffer* partial_dst = one_wg ? state_->out_buf : state_->partial_buf;

  MTL::CommandBuffer* cb = state_->queue->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(state_->active->pso_partial_indexed);
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->grad_buf, 0, 1);
  enc->setBuffer(state_->hess_buf, 0, 2);
  enc->setBuffer(partial_dst, 0, 3);
  enc->setBytes(&nd, sizeof(uint32_t), 4);
  enc->setBytes(&wg, sizeof(uint32_t), 5);
  enc->setBuffer(state_->idx_buf, 0, 6);
  enc->setBytes(&ni, sizeof(uint32_t), 7);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, state_->num_metal_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));

  if (!one_wg) {
    enc->setComputePipelineState(state_->active->pso_reduce);
    enc->setBuffer(state_->partial_buf, 0, 0);
    enc->setBuffer(state_->out_buf, 0, 1);
    enc->setBytes(&wg, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(
        MTL::Size::Make(state_->num_metal_features, 1, 1),
        MTL::Size::Make(state_->active_bins, 1, 1));
  }
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (g_timings.enabled) {
    double gpu_s = cb->GPUEndTime() - cb->GPUStartTime();
    if (gpu_s > 0) g_timings.gpu_us.fetch_add((uint64_t)(gpu_s * 1e6));
  }
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                           bool use_subtract) {
  // Fall back to the CPU path when Metal isn't fully wired or the dataset is
  // ineligible. Also skip the Metal path entirely when quantized gradients
  // are in use — Phase 2.1 doesn't handle the int8/int16 layout yet.
  if (!metal_ready_ || metal_feature_groups_.empty() || config_->use_quantized_grad) {
    // Log the *runtime* fallback reason once per call (Metal init succeeded
    // but per-call gating kicked in). Only logged at Debug; cheap.
    static thread_local bool warned_quantized = false;
    if (config_->use_quantized_grad && metal_ready_ && !warned_quantized) {
      Log::Info("Metal: use_quantized_grad=true is not yet supported on Metal; "
                "delegating histogram construction to CPU for this run.");
      warned_quantized = true;
    }
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }

  Common::FunctionTimer fun_timer("MetalTreeLearner::ConstructHistograms",
                                  global_timer);

  const data_size_t leaf_num_data = smaller_leaf_splits_->num_data_in_leaf();
  const data_size_t* data_indices = smaller_leaf_splits_->data_indices();

  auto t_disp0 = g_timings.enabled ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point();
  if (data_indices == nullptr || leaf_num_data == state_->num_data) {
    // Root-leaf, full-data path: run the un-indexed kernel.
    RunMetalHistogram(gradients_, hessians_, state_->num_data);
  } else {
    // Deeper-leaf path: scatter-gather via the indices buffer on the GPU.
    RunMetalHistogramIndexed(data_indices, leaf_num_data);
  }
  if (g_timings.enabled) {
    auto t1 = std::chrono::steady_clock::now();
    // dispatch_us absorbs idx_copy too — we subtract that below if it ran.
    g_timings.dispatch_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_disp0).count());
    g_timings.calls.fetch_add(1);
  }

  auto t_wb0 = g_timings.enabled ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point();
  const float* metal_hist = static_cast<const float*>(state_->out_buf->contents());
  const int num_features = train_data_->num_features();
  // Per-feature write-back. smaller_leaf_histogram_array_[f].RawData() points
  // at feature f's bin range, with a per-feature `offset` (0 or 1) that
  // tells us how many leading bins the CPU code keeps implicit. The kernel
  // always emits bins 0..num_bin-1; we copy bins offset..num_bin-1 into
  // RawData[0..num_bin - offset - 1].
  // Write-back is bounded by num_features * active_bins memory traffic. For
  // typical tabular sizes the absolute work is ~5-50us; OpenMP fork-join
  // overhead (~5us per parallel region) eats into that. Only parallelize if
  // the work is large enough to amortize the fork-join.
  const bool parallel_writeback = num_features >= 256;
  if (parallel_writeback) {
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < num_features; ++f) {
      if (!is_feature_used[f]) continue;
      const int num_bin = per_feature_num_bin_[f];
      const int offset  = per_feature_offset_[f];
      hist_t* dst = smaller_leaf_histogram_array_[f].RawData();
      const float* src = metal_hist + (size_t)f * state_->active_bins * 2;
      for (int b = offset; b < num_bin; ++b) {
        dst[2 * (b - offset) + 0] = src[2 * b + 0];
        dst[2 * (b - offset) + 1] = src[2 * b + 1];
      }
    }
  } else {
    for (int f = 0; f < num_features; ++f) {
      if (!is_feature_used[f]) continue;
      const int num_bin = per_feature_num_bin_[f];
      const int offset  = per_feature_offset_[f];
      hist_t* dst = smaller_leaf_histogram_array_[f].RawData();
      const float* src = metal_hist + (size_t)f * state_->active_bins * 2;
      for (int b = offset; b < num_bin; ++b) {
        dst[2 * (b - offset) + 0] = src[2 * b + 0];
        dst[2 * (b - offset) + 1] = src[2 * b + 1];
      }
    }
  }
  if (g_timings.enabled) {
    auto t1 = std::chrono::steady_clock::now();
    g_timings.writeback_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_wb0).count());
  }

  // Larger leaf is computed via subtract in the caller when use_subtract=true
  // (typical case). Otherwise (mostly at the root where there's no parent
  // histogram to subtract from), build the larger leaf on Metal too.
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    const data_size_t larger_num_data = larger_leaf_splits_->num_data_in_leaf();
    const data_size_t* larger_indices = larger_leaf_splits_->data_indices();
    if (larger_indices == nullptr || larger_num_data == state_->num_data) {
      RunMetalHistogram(gradients_, hessians_, state_->num_data);
    } else {
      RunMetalHistogramIndexed(larger_indices, larger_num_data);
    }
    const float* metal_hist2 = static_cast<const float*>(state_->out_buf->contents());
    const bool parallel_writeback2 = num_features >= 256;
    auto write_back_larger = [&](int f) {
      if (!is_feature_used[f]) return;
      const int num_bin = per_feature_num_bin_[f];
      const int offset  = per_feature_offset_[f];
      hist_t* dst = larger_leaf_histogram_array_[f].RawData();
      const float* src = metal_hist2 + (size_t)f * state_->active_bins * 2;
      for (int b = offset; b < num_bin; ++b) {
        dst[2 * (b - offset) + 0] = src[2 * b + 0];
        dst[2 * (b - offset) + 1] = src[2 * b + 1];
      }
    };
    if (parallel_writeback2) {
      #pragma omp parallel for schedule(static)
      for (int f = 0; f < num_features; ++f) write_back_larger(f);
    } else {
      for (int f = 0; f < num_features; ++f) write_back_larger(f);
    }
  }
}

}  // namespace LightGBM

#endif  // USE_METAL
