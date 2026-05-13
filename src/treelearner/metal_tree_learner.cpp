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
#include <cstring>
#include <vector>

namespace LightGBM {

namespace {

constexpr int kThreadsPerGroup = 256;
constexpr int kNumBins         = 256;

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
)MSL";

}  // namespace

struct MetalTreeLearner::MetalState {
  MTL::Device*               device  = nullptr;
  MTL::CommandQueue*         queue   = nullptr;
  MTL::Library*              library = nullptr;
  MTL::ComputePipelineState* pso_partial = nullptr;
  MTL::ComputePipelineState* pso_reduce  = nullptr;

  // Persistent device-resident buffers (rebuilt on training-data changes).
  MTL::Buffer* feat_buf     = nullptr;  // uchar [num_metal_features * num_data]
  MTL::Buffer* grad_buf     = nullptr;  // float [num_data]
  MTL::Buffer* hess_buf     = nullptr;  // float [num_data]
  MTL::Buffer* partial_buf  = nullptr;  // float [num_metal_features * wg_per_feat * 2*NUM_BINS]
  MTL::Buffer* out_buf      = nullptr;  // float [num_metal_features * 2*NUM_BINS]

  int num_metal_features = 0;
  int num_data           = 0;
  int wg_per_feat        = 1;

  ~MetalState() {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(feat_buf); rel(grad_buf); rel(hess_buf);
    rel(partial_buf); rel(out_buf);
    rel(pso_partial); rel(pso_reduce);
    rel(library); rel(queue); rel(device);
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
  state_->device = MTL::CreateSystemDefaultDevice();
  if (!state_->device) {
    Log::Warning("Metal: no system default device found. Falling back to CPU path.");
    return;
  }
  Log::Info("Metal device: %s", state_->device->name()->utf8String());

  NS::Error* err = nullptr;
  auto src = NS::String::string(kHistogramKernelSrc, NS::UTF8StringEncoding);
  auto opts = MTL::CompileOptions::alloc()->init();
  state_->library = state_->device->newLibrary(src, opts, &err);
  opts->release();
  if (!state_->library) {
    Log::Warning("Metal: MSL compile failed: %s. Falling back to CPU path.",
                 err ? err->localizedDescription()->utf8String() : "(unknown)");
    return;
  }

  auto make_pso = [&](const char* name) -> MTL::ComputePipelineState* {
    auto nm = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* fn = state_->library->newFunction(nm);
    if (!fn) { Log::Warning("Metal: missing kernel function %s", name); return nullptr; }
    NS::Error* e = nullptr;
    MTL::ComputePipelineState* p = state_->device->newComputePipelineState(fn, &e);
    fn->release();
    if (!p) {
      Log::Warning("Metal: pipeline state creation failed (%s): %s", name,
                   e ? e->localizedDescription()->utf8String() : "(unknown)");
    }
    return p;
  };
  state_->pso_partial = make_pso("histogram_partial");
  state_->pso_reduce  = make_pso("histogram_reduce");
  if (!state_->pso_partial || !state_->pso_reduce) {
    Log::Warning("Metal: pipeline setup incomplete. Falling back to CPU path.");
    return;
  }

  state_->queue = state_->device->newCommandQueue();
  metal_ready_ = true;
  Log::Info("Metal: backend initialized (kernels compiled, queue ready).");
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
  if (num_groups != num_features) {
    Log::Info("Metal: skipping acceleration (multi-feature groups detected).");
    return false;
  }
  for (int g = 0; g < num_groups; ++g) {
    if (train_data_->IsMultiGroup(g)) {
      Log::Info("Metal: skipping acceleration (multi-val feature group %d).", g);
      return false;
    }
    if (train_data_->FeatureGroupNumBin(g) > kNumBins) {
      Log::Info("Metal: skipping acceleration (group %d has %d bins, > %d).",
                g, train_data_->FeatureGroupNumBin(g), kNumBins);
      return false;
    }
  }

  const data_size_t num_data = train_data_->num_data();
  metal_feature_groups_.resize(num_features);
  for (int f = 0; f < num_features; ++f) metal_feature_groups_[f] = f;

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
    for (data_size_t i = 0; i < num_data; ++i) {
      col[i] = static_cast<uint8_t>(it->RawGet(i));
    }
  }

  // Allocate gradient/hessian shared buffers (zero-copy on Apple silicon).
  state_->grad_buf = state_->device->newBuffer((size_t)num_data * sizeof(score_t),
                                               MTL::ResourceStorageModeShared);
  state_->hess_buf = state_->device->newBuffer((size_t)num_data * sizeof(score_t),
                                               MTL::ResourceStorageModeShared);

  // Auto-tune wg_per_feat so a 20-core M-series GPU stays saturated.
  state_->wg_per_feat = std::max(1, std::min(32, 512 / std::max(num_features, 1)));
  state_->num_metal_features = num_features;
  state_->num_data = num_data;

  state_->partial_buf = state_->device->newBuffer(
      (size_t)num_features * (size_t)state_->wg_per_feat * kNumBins * 2 * sizeof(float),
      MTL::ResourceStorageModePrivate);
  state_->out_buf = state_->device->newBuffer(
      (size_t)num_features * kNumBins * 2 * sizeof(float),
      MTL::ResourceStorageModeShared);

  Log::Info("Metal: %d-feature dense buffer materialized (%.1f MB). "
            "wg_per_feat=%d, NUM_BINS=%d.",
            num_features, feat_bytes / 1048576.0, state_->wg_per_feat, kNumBins);
  return true;
}

void MetalTreeLearner::RunMetalHistogram(const score_t* gradients,
                                         const score_t* hessians,
                                         data_size_t num_data) {
  // Copy g/h into shared buffers (zero-copy on Apple silicon).
  std::memcpy(state_->grad_buf->contents(), gradients,
              (size_t)num_data * sizeof(score_t));
  std::memcpy(state_->hess_buf->contents(), hessians,
              (size_t)num_data * sizeof(score_t));

  // Build a tiny constant buffer for num_data + wg_per_feat. These vary per
  // call (num_data shrinks on deeper leaves), so we re-emit each dispatch.
  uint32_t nd = (uint32_t)num_data;
  uint32_t wg = (uint32_t)state_->wg_per_feat;

  MTL::CommandBuffer* cb = state_->queue->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(state_->pso_partial);
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->grad_buf, 0, 1);
  enc->setBuffer(state_->hess_buf, 0, 2);
  enc->setBuffer(state_->partial_buf, 0, 3);
  enc->setBytes(&nd, sizeof(uint32_t), 4);
  enc->setBytes(&wg, sizeof(uint32_t), 5);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, state_->num_metal_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));

  enc->setComputePipelineState(state_->pso_reduce);
  enc->setBuffer(state_->partial_buf, 0, 0);
  enc->setBuffer(state_->out_buf, 0, 1);
  enc->setBytes(&wg, sizeof(uint32_t), 2);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->num_metal_features, 1, 1),
      MTL::Size::Make(kNumBins, 1, 1));
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                           bool use_subtract) {
  // Fall back to the CPU path when Metal isn't fully wired or the dataset is
  // ineligible. Also skip the Metal path entirely when quantized gradients
  // are in use — Phase 2.1 doesn't handle the int8/int16 layout yet.
  if (!metal_ready_ || metal_feature_groups_.empty() || config_->use_quantized_grad) {
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }

  Common::FunctionTimer fun_timer("MetalTreeLearner::ConstructHistograms",
                                  global_timer);

  // Gather ordered gradients/hessians for the smaller leaf. data_indices ==
  // nullptr means "all data in original order" (root case).
  const data_size_t leaf_num_data = smaller_leaf_splits_->num_data_in_leaf();
  const data_size_t* data_indices = smaller_leaf_splits_->data_indices();
  const score_t* g_src = gradients_;
  const score_t* h_src = hessians_;
  if (data_indices != nullptr && leaf_num_data < state_->num_data) {
    score_t* og = ordered_gradients_.data();
    score_t* oh = ordered_hessians_.data();
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < leaf_num_data; ++i) {
      og[i] = gradients_[data_indices[i]];
      oh[i] = hessians_[data_indices[i]];
    }
    OMP_THROW_EX();
    g_src = og;
    h_src = oh;
  }

  // The Metal kernel always iterates the *full* feature column, so it needs
  // the gradient/hessian array indexed by global row id, not leaf row id.
  // For the leaf-subset case, the OpenCL backend passes an `indices` buffer
  // and skips disabled rows. Phase 2.1 doesn't have that yet, so we fall
  // back to the CPU path when this matters (deeper leaves).
  if (leaf_num_data != state_->num_data) {
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }

  // Root-leaf, full-data path: run Metal, then write back into the
  // smaller-leaf histogram array.
  RunMetalHistogram(g_src, h_src, state_->num_data);

  const float* metal_hist = static_cast<const float*>(state_->out_buf->contents());
  hist_t* hist_base =
      smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
  const int num_features = train_data_->num_features();
  #pragma omp parallel for schedule(static)
  for (int f = 0; f < num_features; ++f) {
    if (!is_feature_used[f]) continue;
    const int num_bin = train_data_->FeatureGroupNumBin(f);
    hist_t* dst = hist_base + 2 * train_data_->GroupBinBoundary(f);
    const float* src = metal_hist + (size_t)f * kNumBins * 2;
    for (int b = 0; b < num_bin; ++b) {
      dst[2 * b + 0] = src[2 * b + 0];
      dst[2 * b + 1] = src[2 * b + 1];
    }
  }

  // Larger leaf is computed via subtract in the caller when use_subtract=true.
  // When use_subtract=false (rare at root), build it with the CPU path so we
  // don't have to also reorder gradients for the *other* leaf on GPU yet.
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // We've already produced smaller-leaf histograms; ask the base class to
    // compute only the larger leaf by temporarily zeroing the smaller-leaf
    // marker is brittle. Simpler: delegate the whole thing (smaller leaf gets
    // recomputed on CPU). The double-work is rare and only at split-equal
    // leaf sizes.
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
  }
}

}  // namespace LightGBM

#endif  // USE_METAL
