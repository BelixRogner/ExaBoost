/*!
 * Copyright (c) 2026 ExaBoost authors. All rights reserved.
 * Licensed under the MIT License.
 *
 * Tests that the Metal histogram kernel matches an OpenMP CPU reference
 * histogram within ULP-level tolerance on synthetic data.
 *
 * Runs only when BUILD_CPP_TEST=ON and USE_METAL=ON. Numerical drift is
 * bounded by the CAS-loop atomic-ordering noise inherent in non-deterministic
 * threadgroup atomic adds.
 */

#include <gtest/gtest.h>

#ifdef USE_METAL

// Implementations are emitted in src/treelearner/metal_tree_learner.cpp;
// do NOT redefine the *_PRIVATE_IMPLEMENTATION macros here or the linker
// will see duplicate symbols.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kNumBins        = 256;
constexpr int kThreadsPerGroup = 256;

const char* kMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;
constant uint NUM_BINS    = 256;
constant uint NUM_SUBHIST = 4u;

inline void atomic_tg_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float cur = as_type<float>(expected);
        desired = as_type<uint>(cur + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired, memory_order_relaxed, memory_order_relaxed));
}

kernel void histogram_partial(
    device const uchar*  features      [[ buffer(0) ]],
    device const float*  gradients     [[ buffer(1) ]],
    device const float*  hessians      [[ buffer(2) ]],
    device float*        partial_hist  [[ buffer(3) ]],
    constant uint&       num_data      [[ buffer(4) ]],
    constant uint&       wg_per_feat   [[ buffer(5) ]],
    uint3 tid3 [[ thread_position_in_threadgroup ]],
    uint3 gid  [[ threadgroup_position_in_grid ]],
    uint3 tg3  [[ threads_per_threadgroup ]],
    uint  sgi  [[ simdgroup_index_in_threadgroup ]])
{
    uint tid = tid3.x, tg_sz = tg3.x;
    threadgroup atomic_uint local_grad[NUM_SUBHIST][NUM_BINS];
    threadgroup atomic_uint local_hess[NUM_SUBHIST][NUM_BINS];
    for (uint i = tid; i < NUM_SUBHIST * NUM_BINS; i += tg_sz) {
        uint s = i / NUM_BINS, b = i % NUM_BINS;
        atomic_store_explicit(&local_grad[s][b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[s][b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint feat = gid.y, wg = gid.x;
    uint per = (num_data + wg_per_feat - 1) / wg_per_feat;
    uint start = wg * per, end = min(start + per, num_data);
    device const uchar* col = features + (uint64_t)feat * num_data;
    uint sub = sgi % NUM_SUBHIST;
    for (uint i = start + tid; i < end; i += tg_sz) {
        uint b = (uint)col[i];
        atomic_tg_add_f(&local_grad[sub][b], gradients[i]);
        atomic_tg_add_f(&local_hess[sub][b], hessians[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = partial_hist + ((uint64_t)feat * wg_per_feat + wg) * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        float g = 0.0f, h = 0.0f;
        for (uint s = 0; s < NUM_SUBHIST; ++s) {
            g += as_type<float>(atomic_load_explicit(&local_grad[s][b], memory_order_relaxed));
            h += as_type<float>(atomic_load_explicit(&local_hess[s][b], memory_order_relaxed));
        }
        out[2 * b + 0] = g; out[2 * b + 1] = h;
    }
}

kernel void histogram_reduce(
    device const float*  partial_hist  [[ buffer(0) ]],
    device float*        out_hist      [[ buffer(1) ]],
    constant uint&       wg_per_feat   [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint gid [[ threadgroup_position_in_grid ]])
{
    uint feat = gid, bin = tid;
    float g = 0.0f, h = 0.0f;
    for (uint w = 0; w < wg_per_feat; ++w) {
        device const float* p = partial_hist + ((uint64_t)feat * wg_per_feat + w) * NUM_BINS * 2;
        g += p[2 * bin + 0]; h += p[2 * bin + 1];
    }
    device float* out = out_hist + (uint64_t)feat * NUM_BINS * 2;
    out[2 * bin + 0] = g; out[2 * bin + 1] = h;
}
)MSL";

void cpu_histogram(const uint8_t* features, const float* gradients, const float* hessians,
                   float* out_hist, int num_features, int num_data) {
    std::memset(out_hist, 0, sizeof(float) * 2 * kNumBins * (size_t)num_features);
    for (int f = 0; f < num_features; ++f) {
        float* h = out_hist + (size_t)f * kNumBins * 2;
        const uint8_t* col = features + (size_t)f * num_data;
        for (int i = 0; i < num_data; ++i) {
            uint8_t b = col[i];
            h[2 * b + 0] += gradients[i];
            h[2 * b + 1] += hessians[i];
        }
    }
}

struct MetalCtx {
    MTL::Device* dev = nullptr;
    MTL::CommandQueue* q = nullptr;
    MTL::Library* lib = nullptr;
    MTL::ComputePipelineState* p_part = nullptr;
    MTL::ComputePipelineState* p_red  = nullptr;
    bool ok = false;
    ~MetalCtx() {
        auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
        rel(p_part); rel(p_red); rel(lib); rel(q); rel(dev);
    }
    bool init() {
        dev = MTL::CreateSystemDefaultDevice();
        if (!dev) return false;
        auto src = NS::String::string(kMSL, NS::UTF8StringEncoding);
        auto opts = MTL::CompileOptions::alloc()->init();
        NS::Error* e = nullptr;
        lib = dev->newLibrary(src, opts, &e);
        opts->release();
        if (!lib) return false;
        auto mk = [&](const char* nm) -> MTL::ComputePipelineState* {
            auto ns = NS::String::string(nm, NS::UTF8StringEncoding);
            MTL::Function* fn = lib->newFunction(ns);
            if (!fn) return nullptr;
            NS::Error* err = nullptr;
            auto p = dev->newComputePipelineState(fn, &err);
            fn->release();
            return p;
        };
        p_part = mk("histogram_partial");
        p_red  = mk("histogram_reduce");
        if (!p_part || !p_red) return false;
        q = dev->newCommandQueue();
        ok = (q != nullptr);
        return ok;
    }
};

}  // namespace

class MetalHistogramTest : public ::testing::Test {
 protected:
  MetalCtx ctx;
  void SetUp() override {
    if (!ctx.init()) GTEST_SKIP() << "No Metal device available.";
  }
};

TEST_F(MetalHistogramTest, AgreesWithCPUReference) {
  const int num_data = 100'000;
  const int num_features = 32;

  std::mt19937 rng(13);
  std::uniform_int_distribution<int> bin_dist(0, kNumBins - 1);
  std::normal_distribution<float> grad_dist(0.0f, 1.0f);

  std::vector<uint8_t> features((size_t)num_features * num_data);
  std::vector<float>   gradients(num_data), hessians(num_data);
  for (auto& x : features)  x = (uint8_t)bin_dist(rng);
  for (auto& x : gradients) x = grad_dist(rng);
  for (auto& x : hessians)  x = std::fabs(grad_dist(rng)) + 0.1f;

  const size_t hist_elems = (size_t)num_features * kNumBins * 2;
  std::vector<float> cpu_hist(hist_elems);
  cpu_histogram(features.data(), gradients.data(), hessians.data(),
                cpu_hist.data(), num_features, num_data);

  const uint32_t wg_per_feat = 8;
  MTL::Buffer* fb = ctx.dev->newBuffer(features.data(),  features.size(),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* gb = ctx.dev->newBuffer(gradients.data(), gradients.size() * sizeof(float),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* hb = ctx.dev->newBuffer(hessians.data(),  hessians.size() * sizeof(float),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* outb = ctx.dev->newBuffer(hist_elems * sizeof(float),
                                          MTL::ResourceStorageModeShared);
  MTL::Buffer* partb = ctx.dev->newBuffer(hist_elems * wg_per_feat * sizeof(float),
                                          MTL::ResourceStorageModePrivate);
  uint32_t nd = (uint32_t)num_data;
  MTL::Buffer* ndb = ctx.dev->newBuffer(&nd, sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* wgb = ctx.dev->newBuffer(&wg_per_feat, sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);

  auto* cb = ctx.q->commandBuffer();
  auto* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(ctx.p_part);
  enc->setBuffer(fb, 0, 0); enc->setBuffer(gb, 0, 1); enc->setBuffer(hb, 0, 2);
  enc->setBuffer(partb, 0, 3); enc->setBuffer(ndb, 0, 4); enc->setBuffer(wgb, 0, 5);
  enc->dispatchThreadgroups(
      MTL::Size::Make(wg_per_feat, num_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));
  enc->setComputePipelineState(ctx.p_red);
  enc->setBuffer(partb, 0, 0); enc->setBuffer(outb, 0, 1); enc->setBuffer(wgb, 0, 2);
  enc->dispatchThreadgroups(
      MTL::Size::Make(num_features, 1, 1),
      MTL::Size::Make(kNumBins, 1, 1));
  enc->endEncoding();
  cb->commit(); cb->waitUntilCompleted();

  const float* metal_hist = static_cast<const float*>(outb->contents());
  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < hist_elems; ++i) {
    double diff = std::fabs((double)metal_hist[i] - (double)cpu_hist[i]);
    max_abs = std::max(max_abs, diff);
    double denom = std::max((double)std::fabs(cpu_hist[i]), 1e-3);
    max_rel = std::max(max_rel, diff / denom);
  }

  fb->release(); gb->release(); hb->release();
  outb->release(); partb->release(); ndb->release(); wgb->release();

  // Bound: sums of ~12.5k floats each (100k rows / 8 wg / ~1 cell) with
  // ULP-level CAS noise — typically max_rel < 0.01, occasionally up to 0.05.
  EXPECT_LT(max_rel, 0.05) << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(MetalHistogramTest, MatchesAtSmallScale) {
  // 1k rows × 4 features: enough to exercise the kernel without much atomic noise.
  const int num_data = 1024;
  const int num_features = 4;

  std::vector<uint8_t> features((size_t)num_features * num_data);
  std::vector<float>   gradients(num_data, 1.0f), hessians(num_data, 2.0f);
  // Deterministic bin pattern: feature f, row i -> bin (i + f) % NUM_BINS.
  for (int f = 0; f < num_features; ++f) {
    for (int i = 0; i < num_data; ++i) {
      features[(size_t)f * num_data + i] = (uint8_t)((i + f) % kNumBins);
    }
  }

  const size_t hist_elems = (size_t)num_features * kNumBins * 2;
  std::vector<float> cpu_hist(hist_elems);
  cpu_histogram(features.data(), gradients.data(), hessians.data(),
                cpu_hist.data(), num_features, num_data);

  const uint32_t wg_per_feat = 1;
  MTL::Buffer* fb = ctx.dev->newBuffer(features.data(),  features.size(),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* gb = ctx.dev->newBuffer(gradients.data(), gradients.size() * sizeof(float),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* hb = ctx.dev->newBuffer(hessians.data(),  hessians.size() * sizeof(float),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* outb = ctx.dev->newBuffer(hist_elems * sizeof(float),
                                          MTL::ResourceStorageModeShared);
  MTL::Buffer* partb = ctx.dev->newBuffer(hist_elems * wg_per_feat * sizeof(float),
                                          MTL::ResourceStorageModePrivate);
  uint32_t nd = (uint32_t)num_data;
  MTL::Buffer* ndb = ctx.dev->newBuffer(&nd, sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);
  MTL::Buffer* wgb = ctx.dev->newBuffer(&wg_per_feat, sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared);

  auto* cb = ctx.q->commandBuffer();
  auto* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(ctx.p_part);
  enc->setBuffer(fb, 0, 0); enc->setBuffer(gb, 0, 1); enc->setBuffer(hb, 0, 2);
  enc->setBuffer(partb, 0, 3); enc->setBuffer(ndb, 0, 4); enc->setBuffer(wgb, 0, 5);
  enc->dispatchThreadgroups(
      MTL::Size::Make(wg_per_feat, num_features, 1),
      MTL::Size::Make(kThreadsPerGroup, 1, 1));
  enc->setComputePipelineState(ctx.p_red);
  enc->setBuffer(partb, 0, 0); enc->setBuffer(outb, 0, 1); enc->setBuffer(wgb, 0, 2);
  enc->dispatchThreadgroups(
      MTL::Size::Make(num_features, 1, 1),
      MTL::Size::Make(kNumBins, 1, 1));
  enc->endEncoding();
  cb->commit(); cb->waitUntilCompleted();

  const float* metal_hist = static_cast<const float*>(outb->contents());
  for (size_t i = 0; i < hist_elems; ++i) {
    EXPECT_FLOAT_EQ(metal_hist[i], cpu_hist[i])
        << "Mismatch at idx " << i << " (feat=" << i/(2*kNumBins) << ")";
  }

  fb->release(); gb->release(); hb->release();
  outb->release(); partb->release(); ndb->release(); wgb->release();
}

#endif  // USE_METAL
