// Phase 1 benchmark: gradient/hessian histogram construction.
// Compares Metal (Apple GPU) vs OpenMP CPU baseline. Single-file build.
//
// This mirrors the *essential* work of LightGBM's OpenCL histogram256 kernel
// without the full uchar4 packing / workgroup-per-feature tiling, so the
// per-iteration numbers here are a directional signal for the go/no-go gate,
// not a perf claim for the integrated path.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kNumBins = 256;
constexpr int kThreadsPerGroup = 256;

const char* kMetalKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint NUM_BINS = 256;

// MSL on this system lacks threadgroup atomic_float, so we mirror the
// LightGBM OpenCL pattern: store float histograms as uint bit-patterns in
// threadgroup memory and atomic-add via compare-exchange CAS loop.
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

// One threadgroup per feature. Each thread strides over rows, CAS-adds into
// a threadgroup-local histogram, then writes its slice to device memory.
kernel void histogram_kernel(
    device const uchar*  features    [[ buffer(0) ]],  // [num_features * num_data], features[f*num_data + i]
    device const float*  gradients   [[ buffer(1) ]],  // [num_data]
    device const float*  hessians    [[ buffer(2) ]],  // [num_data]
    device float*        out_hist    [[ buffer(3) ]],  // [num_features * NUM_BINS * 2]
    constant uint&       num_data    [[ buffer(4) ]],
    uint tid    [[ thread_position_in_threadgroup ]],
    uint gid    [[ threadgroup_position_in_grid ]],
    uint tg_sz  [[ threads_per_threadgroup ]])
{
    threadgroup atomic_uint local_grad[NUM_BINS];
    threadgroup atomic_uint local_hess[NUM_BINS];

    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        atomic_store_explicit(&local_grad[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_hess[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar* feat_col = features + (uint64_t)gid * num_data;
    for (uint i = tid; i < num_data; i += tg_sz) {
        uint bin = (uint)feat_col[i];
        atomic_tg_add_f(&local_grad[bin], gradients[i]);
        atomic_tg_add_f(&local_hess[bin], hessians[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out = out_hist + (uint64_t)gid * NUM_BINS * 2;
    for (uint b = tid; b < NUM_BINS; b += tg_sz) {
        out[2 * b + 0] = as_type<float>(atomic_load_explicit(&local_grad[b], memory_order_relaxed));
        out[2 * b + 1] = as_type<float>(atomic_load_explicit(&local_hess[b], memory_order_relaxed));
    }
}
)MSL";

void cpu_histogram(const uint8_t* features, const float* gradients, const float* hessians,
                   float* out_hist, int num_features, int num_data) {
    std::memset(out_hist, 0, sizeof(float) * 2 * kNumBins * (size_t)num_features);
    #pragma omp parallel for schedule(static)
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

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    }
    return m;
}

double max_rel_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double denom = std::max((double)std::fabs(a[i]), 1e-6);
        m = std::max(m, (double)std::fabs(a[i] - b[i]) / denom);
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    int num_data     = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
    int num_features = (argc > 2) ? std::atoi(argv[2]) : 64;
    int iters        = (argc > 3) ? std::atoi(argv[3]) : 20;

    std::printf("Config: num_data=%d, num_features=%d, num_bins=%d, iters=%d\n",
                num_data, num_features, kNumBins, iters);

    // ---- Synthetic data ----
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> bin_dist(0, kNumBins - 1);
    std::normal_distribution<float> grad_dist(0.0f, 1.0f);

    std::vector<uint8_t> features((size_t)num_features * num_data);
    std::vector<float>   gradients(num_data);
    std::vector<float>   hessians(num_data);
    for (auto& x : features)  x = (uint8_t)bin_dist(rng);
    for (auto& x : gradients) x = grad_dist(rng);
    for (auto& x : hessians)  x = std::fabs(grad_dist(rng)) + 0.1f;

    const size_t hist_elems = (size_t)num_features * kNumBins * 2;
    std::vector<float> cpu_hist(hist_elems);

    // ---- CPU benchmark ----
    {
        // Warmup
        cpu_histogram(features.data(), gradients.data(), hessians.data(),
                      cpu_hist.data(), num_features, num_data);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            cpu_histogram(features.data(), gradients.data(), hessians.data(),
                          cpu_hist.data(), num_features, num_data);
        }
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("CPU    : %8.3f ms/iter (total %.1f ms over %d iters)\n",
                    total_ms / iters, total_ms, iters);
    }

    // ---- Metal setup ----
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "No Metal device.\n");
        return 1;
    }
    std::printf("Metal device: %s\n", device->name()->utf8String());

    NS::Error* err = nullptr;
    auto src_nsstring = NS::String::string(kMetalKernelSrc, NS::UTF8StringEncoding);
    auto compile_opts = MTL::CompileOptions::alloc()->init();
    MTL::Library* library = device->newLibrary(src_nsstring, compile_opts, &err);
    compile_opts->release();
    if (!library) {
        std::fprintf(stderr, "MSL compile error: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(null)");
        return 1;
    }

    auto fn_name = NS::String::string("histogram_kernel", NS::UTF8StringEncoding);
    MTL::Function* fn = library->newFunction(fn_name);
    MTL::ComputePipelineState* pso = device->newComputePipelineState(fn, &err);
    if (!pso) {
        std::fprintf(stderr, "Pipeline state error: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(null)");
        return 1;
    }

    MTL::CommandQueue* queue = device->newCommandQueue();

    // Shared-storage buffers — zero-copy on Apple silicon.
    MTL::Buffer* feat_buf = device->newBuffer(features.data(),  features.size(),
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* grad_buf = device->newBuffer(gradients.data(), gradients.size() * sizeof(float),
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* hess_buf = device->newBuffer(hessians.data(),  hessians.size() * sizeof(float),
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* out_buf  = device->newBuffer(hist_elems * sizeof(float),
                                              MTL::ResourceStorageModeShared);
    uint32_t num_data_u = (uint32_t)num_data;
    MTL::Buffer* n_buf   = device->newBuffer(&num_data_u, sizeof(uint32_t),
                                             MTL::ResourceStorageModeShared);

    auto dispatch_one = [&]() {
        MTL::CommandBuffer* cb = queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(feat_buf, 0, 0);
        enc->setBuffer(grad_buf, 0, 1);
        enc->setBuffer(hess_buf, 0, 2);
        enc->setBuffer(out_buf,  0, 3);
        enc->setBuffer(n_buf,    0, 4);
        MTL::Size grid    = MTL::Size::Make(num_features, 1, 1);
        MTL::Size tg_size = MTL::Size::Make(kThreadsPerGroup, 1, 1);
        enc->dispatchThreadgroups(grid, tg_size);
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    };

    // Warmup (also triggers shader cache).
    dispatch_one();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) dispatch_one();
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Metal  : %8.3f ms/iter (total %.1f ms over %d iters)\n",
                total_ms / iters, total_ms, iters);

    // ---- Correctness ----
    const float* gpu_hist = static_cast<const float*>(out_buf->contents());
    double abs_d = max_abs_diff(cpu_hist.data(), gpu_hist, hist_elems);
    double rel_d = max_rel_diff(cpu_hist.data(), gpu_hist, hist_elems);
    std::printf("Diff   : max_abs=%.6f max_rel=%.6f\n", abs_d, rel_d);

    feat_buf->release(); grad_buf->release(); hess_buf->release();
    out_buf->release();  n_buf->release();
    queue->release(); pso->release(); fn->release(); library->release(); device->release();
    pool->release();
    return 0;
}
