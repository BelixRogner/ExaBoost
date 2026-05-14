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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
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

// K=2 multi-feature variant. Each threadgroup processes TWO adjacent features
// sharing one read of (gradient, hessian) per row, halving grad/hess memory
// traffic. Threadgroup memory: 2 * NUM_SUBHIST * NUM_BINS * 2 atomic_uint
// = 16 KB at NUM_BINS=256 (fits comfortably in the 32 KB threadgroup limit).
//
// Opt-in via LIGHTGBM_METAL_K_FEATS=2 — atomic contention may dominate the
// bandwidth savings on some workloads, so this is gated rather than default.
#define K2 2u
kernel void histogram_partial_k2(
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
    threadgroup atomic_uint local_grad[K2][NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_uint local_hess[K2][NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < K2 * NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint k = i / (NUM_SUBHIST * NUM_BINS);
        uint rem = i % (NUM_SUBHIST * NUM_BINS);
        uint s = rem / NUM_BINS, b = rem % NUM_BINS;
        atomic_store_explicit(&local_grad[k][s][b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[k][s][b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat_pair  = gid.y;            // index over (num_features/2) feature pairs
    uint wg_in_feat = gid.x;
    uint feat0      = feat_pair * K2;   // first feature in pair
    uint per_wg     = (num_data + wg_per_feat - 1) / wg_per_feat;
    uint start      = wg_in_feat * per_wg;
    uint end        = min(start + per_wg, num_data);
    device const uchar* col0 = features + (uint64_t)(feat0 + 0) * num_data;
    device const uchar* col1 = features + (uint64_t)(feat0 + 1) * num_data;
    uint sub = sg_idx % NUM_SUBHIST;
    for (uint i = start + tid; i < end; i += tg_sz) {
        float g = gradients[i];
        float h = hessians[i];
        uint b0 = (uint)col0[i];
        uint b1 = (uint)col1[i];
        atomic_tg_add_f(&local_grad[0][sub][b0], g);
        atomic_tg_add_f(&local_hess[0][sub][b0], h);
        atomic_tg_add_f(&local_grad[1][sub][b1], g);
        atomic_tg_add_f(&local_hess[1][sub][b1], h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Write each feature's output to its own slot in partial_hist.
    for (uint k = 0; k < K2; ++k) {
        uint global_feat = feat0 + k;
        device float* out =
            partial_hist + ((uint64_t)global_feat * wg_per_feat + wg_in_feat) * NUM_BINS * 2;
        for (uint b = tid; b < NUM_BINS; b += tg_sz) {
            float g_s = 0.0f, h_s = 0.0f;
            for (uint s = 0; s < NUM_SUBHIST; ++s) {
                g_s += as_type<float>(atomic_load_explicit(&local_grad[k][s][b], memory_order_relaxed));
                h_s += as_type<float>(atomic_load_explicit(&local_hess[k][s][b], memory_order_relaxed));
            }
            out[2 * b + 0] = g_s;
            out[2 * b + 1] = h_s;
        }
    }
}

// ---- Quantized-gradient kernels (use_quantized_grad=true, 32-bit case) ----

// Reads int8 packed grad+hess (byte 2*i = grad, byte 2*i+1 = hess),
// atomic-int32-adds into per-SIMD-group sub-histograms, writes int32 output.
kernel void histogram_partial_q32(
    device const uchar*  features      [[ buffer(0) ]],
    device const char*   gh_packed     [[ buffer(1) ]],   // [2 * num_data] int8s
    device int32_t*      partial_hist  [[ buffer(2) ]],
    constant uint&       num_data      [[ buffer(3) ]],
    constant uint&       wg_per_feat   [[ buffer(4) ]],
    uint3 tid3      [[ thread_position_in_threadgroup ]],
    uint3 gid       [[ threadgroup_position_in_grid ]],
    uint3 tg_sz3    [[ threads_per_threadgroup ]],
    uint  sg_idx    [[ simdgroup_index_in_threadgroup ]])
{
    uint tid = tid3.x, tg_sz = tg_sz3.x;
    threadgroup atomic_int local_grad[NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_int local_hess[NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat_id = gid.y, wg_in_feat = gid.x;
    uint per_wg = (num_data + wg_per_feat - 1) / wg_per_feat;
    uint start = wg_in_feat * per_wg, end = min(start + per_wg, num_data);
    device const uchar* col = features + (uint64_t)feat_id * num_data;
    uint sub = sg_idx % NUM_SUBHIST;
    for (uint i = start + tid; i < end; i += tg_sz) {
        uint bin = (uint)col[i];
        int g = (int)gh_packed[2 * i];      // sign-extends int8 -> int32
        int h = (int)gh_packed[2 * i + 1];
        atomic_fetch_add_explicit(&local_grad[sub][bin], g, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hess[sub][bin], h, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device int32_t* out =
        partial_hist + ((uint64_t)feat_id * wg_per_feat + wg_in_feat) * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        int g_s = 0, h_s = 0;
        for (uint s = 0; s < NUM_SUBHIST; ++s) {
            g_s += atomic_load_explicit(&local_grad[s][b], memory_order_relaxed);
            h_s += atomic_load_explicit(&local_hess[s][b], memory_order_relaxed);
        }
        out[2 * b + 0] = g_s;
        out[2 * b + 1] = h_s;
    }
}

// Indexed quantized variant (deeper leaves).
kernel void histogram_partial_q32_indexed(
    device const uchar*  features      [[ buffer(0) ]],
    device const char*   gh_packed     [[ buffer(1) ]],
    device int32_t*      partial_hist  [[ buffer(2) ]],
    constant uint&       num_data      [[ buffer(3) ]],
    constant uint&       wg_per_feat   [[ buffer(4) ]],
    device const uint*   indices       [[ buffer(5) ]],
    constant uint&       num_idx       [[ buffer(6) ]],
    uint3 tid3      [[ thread_position_in_threadgroup ]],
    uint3 gid       [[ threadgroup_position_in_grid ]],
    uint3 tg_sz3    [[ threads_per_threadgroup ]],
    uint  sg_idx    [[ simdgroup_index_in_threadgroup ]])
{
    uint tid = tid3.x, tg_sz = tg_sz3.x;
    threadgroup atomic_int local_grad[NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_int local_hess[NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat_id = gid.y, wg_in_feat = gid.x;
    uint per_wg = (num_idx + wg_per_feat - 1) / wg_per_feat;
    uint start = wg_in_feat * per_wg, end = min(start + per_wg, num_idx);
    device const uchar* col = features + (uint64_t)feat_id * num_data;
    uint sub = sg_idx % NUM_SUBHIST;
    for (uint j = start + tid; j < end; j += tg_sz) {
        uint row = indices[j];
        uint bin = (uint)col[row];
        int g = (int)gh_packed[2 * row];
        int h = (int)gh_packed[2 * row + 1];
        atomic_fetch_add_explicit(&local_grad[sub][bin], g, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hess[sub][bin], h, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device int32_t* out =
        partial_hist + ((uint64_t)feat_id * wg_per_feat + wg_in_feat) * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        int g_s = 0, h_s = 0;
        for (uint s = 0; s < NUM_SUBHIST; ++s) {
            g_s += atomic_load_explicit(&local_grad[s][b], memory_order_relaxed);
            h_s += atomic_load_explicit(&local_hess[s][b], memory_order_relaxed);
        }
        out[2 * b + 0] = g_s;
        out[2 * b + 1] = h_s;
    }
}

// Quantized reduce: int32 sum across wg_per_feat dim.
kernel void histogram_reduce_q32(
    device const int32_t* partial_hist  [[ buffer(0) ]],
    device int32_t*       out_hist      [[ buffer(1) ]],
    constant uint&        wg_per_feat   [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint gid [[ threadgroup_position_in_grid ]])
{
    uint feat_id = gid;
    uint bin = tid;
    int g = 0, h = 0;
    for (uint w = 0; w < wg_per_feat; ++w) {
        device const int32_t* p = partial_hist + ((uint64_t)feat_id * wg_per_feat + w) * NUM_BINS * 2;
        g += p[2 * bin + 0];
        h += p[2 * bin + 1];
    }
    device int32_t* out = out_hist + (uint64_t)feat_id * NUM_BINS * 2;
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

// Set via LIGHTGBM_METAL_VERIFY=1 — runs CPU histograms alongside Metal,
// asserts agreement within tolerance, logs drift. Catches kernel regressions
// during development. Uses CPU result for actual splits.
struct MetalVerifyState {
  bool enabled = false;
  double max_seen_rel = 0.0;
  uint64_t calls = 0;
  ~MetalVerifyState() {
    if (enabled && calls > 0) {
      std::fprintf(stderr,
        "[metal-verify] %llu Metal vs CPU comparisons; max_seen_rel=%.6f\n",
        (unsigned long long)calls, max_seen_rel);
    }
  }
};
static MetalVerifyState g_verify;

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
    MTL::Library*              library                 = nullptr;
    MTL::ComputePipelineState* pso_partial             = nullptr;
    MTL::ComputePipelineState* pso_partial_indexed     = nullptr;
    MTL::ComputePipelineState* pso_partial_k2          = nullptr;  // multi-feature variant
    MTL::ComputePipelineState* pso_partial_q32         = nullptr;  // quantized 32-bit
    MTL::ComputePipelineState* pso_partial_q32_indexed = nullptr;
    MTL::ComputePipelineState* pso_reduce              = nullptr;
    MTL::ComputePipelineState* pso_reduce_q32          = nullptr;
  };
  PerBinSize bs16, bs64, bs256;

  // Shader-binary archive: persists compiled PSOs to disk so subsequent
  // processes skip the MSL compile. nullptr if disabled or if archive
  // creation fails.
  MTL::BinaryArchive* archive = nullptr;
  bool archive_was_fresh = false;  // true if we created (not loaded) the archive

  PerBinSize* active = nullptr;   // chosen at BuildDenseFeatureBuffer based on max_num_bin
  int active_bins = 0;            // 16 / 64 / 256
  int k_feats = 1;                // 1=default kernel, 2=multi-feature kernel

  // Persistent device-resident buffers (rebuilt on training-data changes).
  MTL::Buffer* feat_buf     = nullptr;  // uchar [num_metal_features * num_data]
  MTL::Buffer* grad_buf     = nullptr;  // float [num_data]
  MTL::Buffer* hess_buf     = nullptr;  // float [num_data]
  MTL::Buffer* idx_buf      = nullptr;  // uint  [num_data]
  MTL::Buffer* idx_buf_larger = nullptr;  // larger-leaf indices for sibling batching
  MTL::Buffer* partial_buf  = nullptr;  // float [num_features * wg_per_feat * 2*active_bins]
  MTL::Buffer* out_buf      = nullptr;  // float [num_features * 2*active_bins]
  // Second output buffer for the sibling-leaf path: when !use_subtract we
  // dispatch BOTH smaller and larger leaf in one command buffer to amortize
  // the waitUntilCompleted overhead. They need distinct destinations so
  // the second kernel doesn't clobber the first.
  MTL::Buffer* out_buf_larger = nullptr;  // float [num_features * 2*active_bins]
  MTL::Buffer* partial_buf_larger = nullptr;
  // Quantized variant buffers (allocated only when use_quantized_grad).
  MTL::Buffer* gh_packed_buf  = nullptr;  // int8  [2 * num_data] packed grad+hess
  MTL::Buffer* partial_buf_i32 = nullptr; // int32 [num_features * wg_per_feat * 2*active_bins]
  MTL::Buffer* out_buf_i32     = nullptr; // int32 [num_features * 2*active_bins]
  bool quantized_buffers = false;

  int num_metal_features = 0;
  int num_data           = 0;
  int wg_per_feat        = 1;

  static void ReleaseBinsize(PerBinSize* bs) {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(bs->pso_partial); rel(bs->pso_partial_indexed); rel(bs->pso_partial_k2);
    rel(bs->pso_partial_q32); rel(bs->pso_partial_q32_indexed);
    rel(bs->pso_reduce); rel(bs->pso_reduce_q32);
    rel(bs->library);
  }

  ~MetalState() {
    auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
    rel(feat_buf); rel(grad_buf); rel(hess_buf); rel(idx_buf); rel(idx_buf_larger);
    rel(partial_buf); rel(out_buf);
    rel(out_buf_larger); rel(partial_buf_larger);
    rel(gh_packed_buf); rel(partial_buf_i32); rel(out_buf_i32);
    ReleaseBinsize(&bs16); ReleaseBinsize(&bs64); ReleaseBinsize(&bs256);
    rel(archive);
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
  metal_buffer_ready_ = false;
}

void MetalTreeLearner::InitMetal() {
  // Honor LIGHTGBM_METAL_TIMING + LIGHTGBM_METAL_VERIFY once globally.
  if (const char* env = std::getenv("LIGHTGBM_METAL_TIMING")) {
    if (env[0] == '1') g_timings.enabled = true;
  }
  if (const char* env = std::getenv("LIGHTGBM_METAL_VERIFY")) {
    if (env[0] == '1') {
      g_verify.enabled = true;
      Log::Info("Metal: LIGHTGBM_METAL_VERIFY=1 — running CPU alongside Metal "
                "to catch drift. Final histograms will be the CPU values.");
    }
  }

  state_->device = MTL::CreateSystemDefaultDevice();
  if (!state_->device) {
    Log::Warning("Metal: no system default device found. Falling back to CPU path.");
    return;
  }
  Log::Info("Metal device: %s", state_->device->name()->utf8String());

  // Shader-binary archive: load existing if present, otherwise create fresh
  // (we'll populate during PSO compilation and serialize at end of Init).
  // Disable via LIGHTGBM_METAL_NO_BINARY_ARCHIVE=1 (e.g., if the cache
  // is somehow corrupt and you want to force a recompile).
  bool archive_enabled = true;
  if (const char* env = std::getenv("LIGHTGBM_METAL_NO_BINARY_ARCHIVE")) {
    if (env[0] == '1') archive_enabled = false;
  }
  std::string archive_path;
  if (archive_enabled) {
    const char* home = std::getenv("HOME");
    if (home) {
      archive_path = std::string(home) + "/.cache/exaboost";
      // mkdir -p (no error if exists).
      mkdir(archive_path.c_str(), 0755);
      // Append a version suffix; bump when kernel source changes.
      archive_path += "/metal_kernels_v1.binarchive";

      auto desc = MTL::BinaryArchiveDescriptor::alloc()->init();
      // Try to load existing file; if missing, leave url unset to create fresh.
      struct stat st;
      bool exists = (stat(archive_path.c_str(), &st) == 0);
      if (exists) {
        auto url = NS::URL::fileURLWithPath(
            NS::String::string(archive_path.c_str(), NS::UTF8StringEncoding));
        desc->setUrl(url);
      }
      NS::Error* err = nullptr;
      state_->archive = state_->device->newBinaryArchive(desc, &err);
      desc->release();
      state_->archive_was_fresh = !exists;
      if (!state_->archive) {
        Log::Info("Metal: BinaryArchive %s unavailable (%s); will recompile.",
                  exists ? "load" : "create",
                  err ? err->localizedDescription()->utf8String() : "(unknown)");
      } else {
        Log::Info("Metal: BinaryArchive %s (%s).",
                  exists ? "loaded" : "fresh", archive_path.c_str());
      }
    }
  }

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
      MTL::ComputePipelineState* p = nullptr;
      if (state_->archive) {
        // Use a ComputePipelineDescriptor with the archive attached so Metal
        // reuses cached binaries or registers a freshly-compiled PSO.
        auto pso_desc = MTL::ComputePipelineDescriptor::alloc()->init();
        pso_desc->setComputeFunction(fn);
        const MTL::BinaryArchive* arcs[1] = { state_->archive };
        auto arr = NS::Array::array((const NS::Object* const*)arcs, 1);
        pso_desc->setBinaryArchives(arr);
        p = state_->device->newComputePipelineState(
            pso_desc, MTL::PipelineOptionNone, nullptr, &err2);
        if (p && state_->archive_was_fresh) {
          NS::Error* add_err = nullptr;
          state_->archive->addComputePipelineFunctions(pso_desc, &add_err);
          // Non-fatal if add fails; we'll just not cache this one.
        }
        pso_desc->release();
      } else {
        p = state_->device->newComputePipelineState(fn, &err2);
      }
      fn->release();
      if (!p) Log::Warning("Metal: PSO creation failed (%s, NUM_BINS=%d): %s",
                           name, num_bins,
                           err2 ? err2->localizedDescription()->utf8String() : "(unknown)");
      return p;
    };
    bs->pso_partial             = make_pso("histogram_partial");
    bs->pso_partial_indexed     = make_pso("histogram_partial_indexed");
    bs->pso_partial_k2          = make_pso("histogram_partial_k2");
    bs->pso_partial_q32         = make_pso("histogram_partial_q32");
    bs->pso_partial_q32_indexed = make_pso("histogram_partial_q32_indexed");
    bs->pso_reduce              = make_pso("histogram_reduce");
    bs->pso_reduce_q32          = make_pso("histogram_reduce_q32");
    return bs->pso_partial && bs->pso_partial_indexed && bs->pso_partial_k2
        && bs->pso_partial_q32 && bs->pso_partial_q32_indexed
        && bs->pso_reduce && bs->pso_reduce_q32;
  };
  bool ok = compile_variant(&state_->bs16,  16,  16)
         && compile_variant(&state_->bs64,  64,  8)
         && compile_variant(&state_->bs256, 256, 4);
  if (!ok) {
    Log::Warning("Metal: kernel compile incomplete. Falling back to CPU path.");
    return;
  }

  // Serialize the archive to disk if we just populated it. Subsequent
  // processes will load from this file and skip the MSL compile.
  if (state_->archive && state_->archive_was_fresh && !archive_path.empty()) {
    auto url = NS::URL::fileURLWithPath(
        NS::String::string(archive_path.c_str(), NS::UTF8StringEncoding));
    NS::Error* serr = nullptr;
    bool ok2 = state_->archive->serializeToURL(url, &serr);
    if (!ok2) {
      Log::Info("Metal: BinaryArchive save to %s failed: %s",
                archive_path.c_str(),
                serr ? serr->localizedDescription()->utf8String() : "(unknown)");
    } else {
      Log::Info("Metal: BinaryArchive saved to %s.", archive_path.c_str());
    }
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
  // Eligibility check for the Metal histogram path. Multi-feature groups
  // and multi-val (sparse) groups are also supported (the latter opt-in via
  // LIGHTGBM_METAL_FORCE_MULTI_VAL=1 since CPU's sparse path is typically
  // faster). For all eligible features we materialize a packed
  // [num_features × num_data] uchar buffer once at Init.
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
  // per-feature column buffer. Multi-val sparse groups: BinIterator::Get
  // does return per-sub-feature bins correctly, but the dense-buffer
  // materialization loses CPU's sparse-optimized path. Benchmarks show CPU
  // wins ~1.5x on sparse data, so by default we skip multi-val groups and
  // delegate to the CPU path. Opt in via LIGHTGBM_METAL_FORCE_MULTI_VAL=1.
  // Read the env var fresh on each BuildDenseFeatureBuffer (rather than
  // statically once at process start), so tests can toggle it between
  // training runs in the same process.
  bool force_multi_val = false;
  if (const char* env = std::getenv("LIGHTGBM_METAL_FORCE_MULTI_VAL")) {
    if (env[0] == '1') force_multi_val = true;
  }
  for (int g = 0; g < num_groups; ++g) {
    if (train_data_->IsMultiGroup(g) && !force_multi_val) {
      Log::Info("Metal: skipping acceleration (multi-val feature group %d; "
                "CPU's sparse path is typically faster. Override with "
                "LIGHTGBM_METAL_FORCE_MULTI_VAL=1).", g);
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
  per_feature_num_bin_.resize(num_features);
  per_feature_offset_.resize(num_features);
  for (int f = 0; f < num_features; ++f) {
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
  // Multi-feature kernel: opt in via LIGHTGBM_METAL_K_FEATS=2. Only safe when
  // num_features is even and the active binsize fits 2x in threadgroup memory.
  state_->k_feats = 1;
  if (const char* env = std::getenv("LIGHTGBM_METAL_K_FEATS")) {
    int v = std::atoi(env);
    if (v == 2 && (num_features % 2 == 0)) {
      state_->k_feats = 2;
      Log::Info("Metal: K_FEATS=2 enabled (multi-feature kernel).");
    } else if (v == 2) {
      Log::Info("Metal: K_FEATS=2 requested but num_features=%d is odd; falling "
                "back to K=1.", num_features);
    }
  }
  state_->num_metal_features = num_features;
  state_->num_data = num_data;

  state_->partial_buf = state_->device->newBuffer(
      (size_t)num_features * (size_t)state_->wg_per_feat * state_->active_bins * 2 * sizeof(float),
      MTL::ResourceStorageModePrivate);
  state_->out_buf = state_->device->newBuffer(
      (size_t)num_features * state_->active_bins * 2 * sizeof(float),
      MTL::ResourceStorageModeShared);
  state_->out_buf_larger = state_->device->newBuffer(
      (size_t)num_features * state_->active_bins * 2 * sizeof(float),
      MTL::ResourceStorageModeShared);
  state_->partial_buf_larger = state_->device->newBuffer(
      (size_t)num_features * (size_t)state_->wg_per_feat * state_->active_bins * 2 * sizeof(float),
      MTL::ResourceStorageModePrivate);
  // Pre-allocate the indices buffer to avoid per-call allocation. Sized for
  // the worst case (all rows in a leaf).
  state_->idx_buf = state_->device->newBuffer((size_t)num_data * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
  state_->idx_buf_larger = state_->device->newBuffer((size_t)num_data * sizeof(uint32_t),
                                                      MTL::ResourceStorageModeShared);
  // Quantized buffers when use_quantized_grad is enabled. Same shape as the
  // float buffers but int32-element-typed.
  if (config_->use_quantized_grad) {
    state_->gh_packed_buf = state_->device->newBuffer(
        (size_t)num_data * 2,    // 2 int8s per row
        MTL::ResourceStorageModeShared);
    state_->partial_buf_i32 = state_->device->newBuffer(
        (size_t)num_features * (size_t)state_->wg_per_feat * state_->active_bins * 2 * sizeof(int32_t),
        MTL::ResourceStorageModePrivate);
    state_->out_buf_i32 = state_->device->newBuffer(
        (size_t)num_features * state_->active_bins * 2 * sizeof(int32_t),
        MTL::ResourceStorageModeShared);
    state_->quantized_buffers = true;
    Log::Info("Metal: quantized int32 buffers allocated (%.1f MB).",
              ((size_t)num_features * (size_t)state_->wg_per_feat * state_->active_bins * 2 * sizeof(int32_t)
               + (size_t)num_features * state_->active_bins * 2 * sizeof(int32_t)
               + (size_t)num_data * 2) / 1048576.0);
  }

  Log::Info("Metal: %d-feature dense buffer materialized (%.1f MB). "
            "wg_per_feat=%d, NUM_BINS=%d.",
            num_features, feat_bytes / 1048576.0, state_->wg_per_feat,
            state_->active_bins);
  metal_buffer_ready_ = true;
  return true;
}

void MetalTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  // Gradients/hessians are set once per tree by the caller before BeforeTrain.
  // Copy them into the shared Metal buffers exactly once here so the per-leaf
  // histogram dispatch doesn't re-pay the memcpy.
  if (metal_ready_ && !!metal_buffer_ready_) {
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
  // Choose partial-kernel variant: K=2 multi-feature when enabled, otherwise K=1.
  if (state_->k_feats == 2) {
    enc->setComputePipelineState(state_->active->pso_partial_k2);
  } else {
    enc->setComputePipelineState(state_->active->pso_partial);
  }
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->grad_buf, 0, 1);
  enc->setBuffer(state_->hess_buf, 0, 2);
  enc->setBuffer(partial_dst, 0, 3);
  enc->setBytes(&nd, sizeof(uint32_t), 4);
  enc->setBytes(&wg, sizeof(uint32_t), 5);
  // K=2 packs 2 features per threadgroup, halving the y-dim grid count.
  uint feat_grid = (uint)(state_->num_metal_features / state_->k_feats);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, feat_grid, 1),
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

void MetalTreeLearner::RunMetalHistogramSibling(
    const data_size_t* smaller_indices, data_size_t smaller_n,
    const data_size_t* larger_indices,  data_size_t larger_n) {
  // grad/hess are in shared buffers (BeforeTrain). Stage both index arrays.
  static_assert(sizeof(data_size_t) == sizeof(uint32_t),
                "Metal sibling kernel assumes data_size_t is 32-bit.");
  const bool smaller_full = (smaller_indices == nullptr || smaller_n == state_->num_data);
  const bool larger_full  = (larger_indices  == nullptr || larger_n  == state_->num_data);
  if (!smaller_full) {
    std::memcpy(state_->idx_buf->contents(), smaller_indices,
                (size_t)smaller_n * sizeof(uint32_t));
  }
  if (!larger_full) {
    std::memcpy(state_->idx_buf_larger->contents(), larger_indices,
                (size_t)larger_n * sizeof(uint32_t));
  }

  uint32_t nd = (uint32_t)state_->num_data;
  uint32_t wg = (uint32_t)state_->wg_per_feat;
  uint32_t ns = (uint32_t)smaller_n;
  uint32_t nl = (uint32_t)larger_n;
  const bool one_wg = (state_->wg_per_feat == 1);
  MTL::Buffer* part_dst_s = one_wg ? state_->out_buf        : state_->partial_buf;
  MTL::Buffer* part_dst_l = one_wg ? state_->out_buf_larger : state_->partial_buf_larger;

  MTL::CommandBuffer* cb = state_->queue->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

  // --- Smaller leaf ---
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->grad_buf, 0, 1);
  enc->setBuffer(state_->hess_buf, 0, 2);
  enc->setBuffer(part_dst_s, 0, 3);
  enc->setBytes(&nd, sizeof(uint32_t), 4);
  enc->setBytes(&wg, sizeof(uint32_t), 5);
  if (smaller_full) {
    enc->setComputePipelineState(state_->active->pso_partial);
  } else {
    enc->setComputePipelineState(state_->active->pso_partial_indexed);
    enc->setBuffer(state_->idx_buf, 0, 6);
    enc->setBytes(&ns, sizeof(uint32_t), 7);
  }
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

  // --- Larger leaf (writes to a SEPARATE out_buf so smaller's data isn't clobbered) ---
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->grad_buf, 0, 1);
  enc->setBuffer(state_->hess_buf, 0, 2);
  enc->setBuffer(part_dst_l, 0, 3);
  enc->setBytes(&nd, sizeof(uint32_t), 4);
  enc->setBytes(&wg, sizeof(uint32_t), 5);
  if (larger_full) {
    enc->setComputePipelineState(state_->active->pso_partial);
  } else {
    enc->setComputePipelineState(state_->active->pso_partial_indexed);
    enc->setBuffer(state_->idx_buf_larger, 0, 6);
    enc->setBytes(&nl, sizeof(uint32_t), 7);
  }
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, state_->num_metal_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));
  if (!one_wg) {
    enc->setComputePipelineState(state_->active->pso_reduce);
    enc->setBuffer(state_->partial_buf_larger, 0, 0);
    enc->setBuffer(state_->out_buf_larger, 0, 1);
    enc->setBytes(&wg, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(
        MTL::Size::Make(state_->num_metal_features, 1, 1),
        MTL::Size::Make(state_->active_bins, 1, 1));
  }

  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();  // ONE wait for both leaves
  if (g_timings.enabled) {
    double gpu_s = cb->GPUEndTime() - cb->GPUStartTime();
    if (gpu_s > 0) g_timings.gpu_us.fetch_add((uint64_t)(gpu_s * 1e6));
  }
}

void MetalTreeLearner::RunMetalHistogramQ32(data_size_t num_data) {
  uint32_t nd = (uint32_t)num_data;
  uint32_t wg = (uint32_t)state_->wg_per_feat;
  const bool one_wg = (state_->wg_per_feat == 1);
  MTL::Buffer* partial_dst = one_wg ? state_->out_buf_i32 : state_->partial_buf_i32;

  MTL::CommandBuffer* cb = state_->queue->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(state_->active->pso_partial_q32);
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->gh_packed_buf, 0, 1);
  enc->setBuffer(partial_dst, 0, 2);
  enc->setBytes(&nd, sizeof(uint32_t), 3);
  enc->setBytes(&wg, sizeof(uint32_t), 4);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, state_->num_metal_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));
  if (!one_wg) {
    enc->setComputePipelineState(state_->active->pso_reduce_q32);
    enc->setBuffer(state_->partial_buf_i32, 0, 0);
    enc->setBuffer(state_->out_buf_i32, 0, 1);
    enc->setBytes(&wg, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(
        MTL::Size::Make(state_->num_metal_features, 1, 1),
        MTL::Size::Make(state_->active_bins, 1, 1));
  }
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
}

void MetalTreeLearner::RunMetalHistogramQ32Indexed(const data_size_t* data_indices,
                                                   data_size_t num_idx) {
  static_assert(sizeof(data_size_t) == sizeof(uint32_t),
                "Metal q32 indexed kernel assumes data_size_t is 32-bit.");
  std::memcpy(state_->idx_buf->contents(), data_indices,
              (size_t)num_idx * sizeof(uint32_t));
  uint32_t nd = (uint32_t)state_->num_data;
  uint32_t ni = (uint32_t)num_idx;
  uint32_t wg = (uint32_t)state_->wg_per_feat;
  const bool one_wg = (state_->wg_per_feat == 1);
  MTL::Buffer* partial_dst = one_wg ? state_->out_buf_i32 : state_->partial_buf_i32;

  MTL::CommandBuffer* cb = state_->queue->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(state_->active->pso_partial_q32_indexed);
  enc->setBuffer(state_->feat_buf, 0, 0);
  enc->setBuffer(state_->gh_packed_buf, 0, 1);
  enc->setBuffer(partial_dst, 0, 2);
  enc->setBytes(&nd, sizeof(uint32_t), 3);
  enc->setBytes(&wg, sizeof(uint32_t), 4);
  enc->setBuffer(state_->idx_buf, 0, 5);
  enc->setBytes(&ni, sizeof(uint32_t), 6);
  enc->dispatchThreadgroups(
      MTL::Size::Make(state_->wg_per_feat, state_->num_metal_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));
  if (!one_wg) {
    enc->setComputePipelineState(state_->active->pso_reduce_q32);
    enc->setBuffer(state_->partial_buf_i32, 0, 0);
    enc->setBuffer(state_->out_buf_i32, 0, 1);
    enc->setBytes(&wg, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(
        MTL::Size::Make(state_->num_metal_features, 1, 1),
        MTL::Size::Make(state_->active_bins, 1, 1));
  }
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                           bool use_subtract) {
  // Fall back to the CPU path when Metal isn't fully wired or the dataset is
  // ineligible.
  if (!metal_ready_ || !metal_buffer_ready_) {
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }

  // Quantized-gradient path: int8 grad+hess, int32 histograms. Only the
  // 32-bit-per-bin case is Metal-accelerated; for 16-bit leaves (small
  // leaves), fall back to the CPU path for this call. Verify mode also
  // falls back since it requires float histograms for comparison.
  if (config_->use_quantized_grad) {
    if (g_verify.enabled || !state_->quantized_buffers) {
      SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
      return;
    }
    const uint8_t smaller_bits = gradient_discretizer_->GetHistBitsInLeaf<false>(
        smaller_leaf_splits_->leaf_index());
    if (smaller_bits != 32) {
      // 16-bit leaf — Metal q32 doesn't apply; delegate.
      SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
      return;
    }
    Common::FunctionTimer fun_timer_q("MetalTreeLearner::ConstructHistograms[q32]",
                                       global_timer);

    // Stage the int8 packed gradients/hessians into the GPU buffer.
    const int8_t* gh = gradient_discretizer_->discretized_gradients_and_hessians();
    std::memcpy(state_->gh_packed_buf->contents(), gh,
                (size_t)state_->num_data * 2);

    const data_size_t leaf_num_data_q = smaller_leaf_splits_->num_data_in_leaf();
    const data_size_t* data_indices_q = smaller_leaf_splits_->data_indices();
    if (data_indices_q == nullptr || leaf_num_data_q == state_->num_data) {
      RunMetalHistogramQ32(state_->num_data);
    } else {
      RunMetalHistogramQ32Indexed(data_indices_q, leaf_num_data_q);
    }

    // Write back as int32 to RawDataInt32 — per-feature.
    const int32_t* metal_hist_i32 =
        static_cast<const int32_t*>(state_->out_buf_i32->contents());
    const int num_features = train_data_->num_features();
    for (int f = 0; f < num_features; ++f) {
      if (!is_feature_used[f]) continue;
      const int num_bin = per_feature_num_bin_[f];
      const int offset  = per_feature_offset_[f];
      int32_t* dst = smaller_leaf_histogram_array_[f].RawDataInt32();
      const int32_t* src = metal_hist_i32 + (size_t)f * state_->active_bins * 2;
      for (int b = offset; b < num_bin; ++b) {
        dst[2 * (b - offset) + 0] = src[2 * b + 0];
        dst[2 * (b - offset) + 1] = src[2 * b + 1];
      }
    }
    // Larger leaf is computed via subtract in the caller when use_subtract=true;
    // otherwise delegate to CPU (we don't yet build the larger leaf in q32 mode).
    if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
      SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    }
    return;
  }

  Common::FunctionTimer fun_timer("MetalTreeLearner::ConstructHistograms",
                                  global_timer);

  const data_size_t leaf_num_data = smaller_leaf_splits_->num_data_in_leaf();
  const data_size_t* data_indices = smaller_leaf_splits_->data_indices();

  // No leaf-size threshold: empirically a 600us Metal dispatch beats CPU
  // even for very small leaves (depth-N leaves of a num_data row split
  // typically still have num_data/2^N rows, and CPU writes to many feature
  // histograms which adds up). Users can override via
  // LIGHTGBM_METAL_MIN_LEAF_ROWS=N to opt out at large leaf sizes.
  static int small_leaf_threshold = []() {
    const char* env = std::getenv("LIGHTGBM_METAL_MIN_LEAF_ROWS");
    return env ? std::max(0, std::atoi(env)) : 0;
  }();
  if (small_leaf_threshold > 0 && leaf_num_data < small_leaf_threshold) {
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }

  auto t_disp0 = g_timings.enabled ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point();
  // Sibling-batched path: when we need BOTH leaves' histograms (use_subtract
  // is false and there's a larger leaf), dispatch both in one command
  // buffer to amortize the waitUntilCompleted overhead. ~one wait instead
  // of two.
  const bool sibling_batched =
      !use_subtract && larger_leaf_histogram_array_ != nullptr &&
      larger_leaf_splits_ != nullptr && larger_leaf_splits_->leaf_index() >= 0;
  if (sibling_batched) {
    const data_size_t larger_num_data = larger_leaf_splits_->num_data_in_leaf();
    const data_size_t* larger_indices = larger_leaf_splits_->data_indices();
    RunMetalHistogramSibling(data_indices, leaf_num_data,
                             larger_indices, larger_num_data);
  } else if (data_indices == nullptr || leaf_num_data == state_->num_data) {
    // Root-leaf, full-data path: run the un-indexed kernel.
    RunMetalHistogram(gradients_, hessians_, state_->num_data);
  } else {
    // Deeper-leaf path: scatter-gather via the indices buffer on the GPU.
    RunMetalHistogramIndexed(data_indices, leaf_num_data);
  }
  if (g_timings.enabled) {
    auto t1 = std::chrono::steady_clock::now();
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
  // the work is large enough to amortize the fork-join. Override via env
  // var if your dataset shape needs a different cutoff.
  static int writeback_omp_threshold = []() {
    const char* env = std::getenv("LIGHTGBM_METAL_OMP_WRITEBACK_THRESHOLD");
    return env ? std::max(1, std::atoi(env)) : 256;
  }();
  const bool parallel_writeback = num_features >= writeback_omp_threshold;
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

  // Verify mode: snapshot Metal histograms, run CPU, compare, log drift.
  // CPU values then become the source of truth for the rest of training.
  if (g_verify.enabled) {
    const int num_features = train_data_->num_features();
    std::vector<std::vector<float>> metal_snapshot(num_features);
    for (int f = 0; f < num_features; ++f) {
      if (!is_feature_used[f]) continue;
      const int n = per_feature_num_bin_[f] - per_feature_offset_[f];
      hist_t* d = smaller_leaf_histogram_array_[f].RawData();
      metal_snapshot[f].assign(d, d + 2 * n);
    }
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    double max_rel = 0.0;
    for (int f = 0; f < num_features; ++f) {
      if (!is_feature_used[f]) continue;
      const hist_t* cpu = smaller_leaf_histogram_array_[f].RawData();
      const float* metal = metal_snapshot[f].data();
      const int n = per_feature_num_bin_[f] - per_feature_offset_[f];
      for (int i = 0; i < 2 * n; ++i) {
        double diff = std::fabs((double)metal[i] - (double)cpu[i]);
        double denom = std::max(1e-3, std::fabs((double)cpu[i]));
        max_rel = std::max(max_rel, diff / denom);
      }
    }
    g_verify.max_seen_rel = std::max(g_verify.max_seen_rel, max_rel);
    g_verify.calls++;
    if (max_rel > 0.1) {
      Log::Warning("Metal: verify drift max_rel=%.4f on smaller leaf "
                   "(features=%d)", max_rel, num_features);
    }
    return;  // CPU already produced larger leaf if needed.
  }

  // Larger-leaf path: when use_subtract=true, the caller computes the
  // larger histogram via the subtract trick. When sibling_batched fired,
  // the larger leaf's histogram is already in out_buf_larger from the
  // batched dispatch — just write it back.
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    const float* metal_hist2 = sibling_batched
        ? static_cast<const float*>(state_->out_buf_larger->contents())
        : nullptr;
    if (!sibling_batched) {
      const data_size_t larger_num_data = larger_leaf_splits_->num_data_in_leaf();
      const data_size_t* larger_indices = larger_leaf_splits_->data_indices();
      if (larger_indices == nullptr || larger_num_data == state_->num_data) {
        RunMetalHistogram(gradients_, hessians_, state_->num_data);
      } else {
        RunMetalHistogramIndexed(larger_indices, larger_num_data);
      }
      metal_hist2 = static_cast<const float*>(state_->out_buf->contents());
    }
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
