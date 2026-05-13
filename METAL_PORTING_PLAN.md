# Metal Backend Porting Plan

Goal: add a Metal-based GPU backend (`device_type=metal`) to ExaBoost so training
on Apple silicon can use the GPU. The OpenCL backend stays untouched.

## Scope (Phase 2 target)

Mirror the **OpenCL** backend (`USE_GPU`), not CUDA. The OpenCL backend only
accelerates **dense feature histogram construction**; everything else runs on
CPU. CUDA is a full pipeline rewrite (best-split, data partition, leaf splits,
gradient discretizer, single-GPU tree learner) — out of scope for this branch.

## Inventory of what to port

### Kernels (`src/treelearner/ocl/` → `src/treelearner/metal/`)

| OpenCL file | Metal file | LOC | Variants |
|---|---|---|---|
| `histogram16.cl`  | `histogram16.metal`  | 766 | 16-bin features  |
| `histogram64.cl`  | `histogram64.metal`  | 742 | 64-bin features  |
| `histogram256.cl` | `histogram256.metal` | 791 | 256-bin features |

Each kernel is compiled at runtime with three flavor toggles plus a workgroup
exponent, producing `3 × 3 × (kMaxLogWorkgroupsPerFeature+1)` pipeline states:

- with-indices + masked features (`histogram_kernels_`)
- with-indices + all features (`histogram_allfeats_kernels_`)
- no-indices (root) + all features (`histogram_fulldata_kernels_`)

### Host wrapper

| OpenCL file | Metal file |
|---|---|
| `gpu_tree_learner.h`   | `metal_tree_learner.h`   |
| `gpu_tree_learner.cpp` | `metal_tree_learner.cpp` |

### Build + dispatch

- `CMakeLists.txt`: add `option(USE_METAL ...)`, gate on `APPLE`, link
  `"-framework Metal" "-framework Foundation" "-framework QuartzCore"`, include
  `metal-cpp` headers under `external_libs/metal-cpp/`.
- `src/treelearner/tree_learner.cpp::CreateTreeLearner`: new
  `device_type == "metal"` branch that returns `MetalTreeLearner`.
- `include/LightGBM/config.h`: document `device_type=metal`.

## OpenCL → Metal translation table

| OpenCL                              | Metal Shading Language (MSL)                              |
|---|---|
| `__kernel void f(...)`              | `kernel void f(...)`                                      |
| `__global T*`                       | `device T*`                                               |
| `__local T*`                        | `threadgroup T*`                                          |
| `__constant T*`                     | `constant T*`                                             |
| `get_local_id(0)`                   | `[[thread_position_in_threadgroup]]` attr                 |
| `get_global_id(0)`                  | `[[thread_position_in_grid]]` attr                        |
| `get_group_id(0)`                   | `[[threadgroup_position_in_grid]]` attr                   |
| `get_local_size(0)`                 | `[[threads_per_threadgroup]]` attr                        |
| `barrier(CLK_LOCAL_MEM_FENCE)`      | `threadgroup_barrier(mem_flags::mem_threadgroup)`         |
| `atom_cmpxchg` on `__local uint*`   | `atomic_compare_exchange_weak_explicit` on `threadgroup atomic_uint*` |
| float-add CAS loop                  | `atomic_fetch_add_explicit` on `threadgroup atomic_float*` (MSL 3.0+, Apple silicon) |
| `uchar4`, `float2`                  | same names in MSL                                         |
| `restrict`                          | drop (or `__restrict__` if supported)                     |
| `as_uint`, `as_float`               | `as_type<uint>(x)`, `as_type<float>(x)`                   |
| OpenCL extension pragmas            | drop entirely                                             |

### Atomic float-add — important simplification

The OpenCL kernels use an open-coded CAS loop on `as_acc_int_type` because
OpenCL has no native float atomics. **MSL 3.0+ (macOS 13+, Apple silicon) has
native `atomic_float` for threadgroup memory.** We should use it:

```cpp
threadgroup atomic_float* a = ...;
atomic_fetch_add_explicit(a, val, memory_order_relaxed);
```

This is faster and simpler. Keep a CAS-loop fallback only if we need Intel-Mac
support — and we probably don't, since Apple silicon is the whole point.

## Boost.Compute → metal-cpp host-side translation

| Boost.Compute                              | metal-cpp                                                |
|---|---|
| `boost::compute::context`                  | `MTL::Device*` (Metal has no separate context)           |
| `boost::compute::command_queue`            | `MTL::CommandQueue*`                                     |
| `boost::compute::buffer`                   | `MTL::Buffer*`                                           |
| `boost::compute::vector<T>(n, ctx)`        | `device->newBuffer(n*sizeof(T), MTL::ResourceStorageMode...)` |
| `program::build_with_source(src, ctx, opts)` | `device->newLibrary(NS::String*, MTL::CompileOptions*)` (opts via `preprocessorMacros`) |
| `kernel` from program                      | `MTL::Function*` + `MTL::ComputePipelineState*`           |
| `kernel.set_arg(i, buf)`                   | `encoder->setBuffer(buf, 0, i)`                          |
| `queue.enqueue_1d_range_kernel(...)`       | `encoder->dispatchThreadgroups(..., MTL::Size(...))`     |
| `queue.enqueue_map_buffer(...)`            | unified memory: `buffer->contents()` (no map needed)     |
| `boost::compute::wait_list` + `event`      | `MTL::CommandBuffer*` + `commandBuffer->waitUntilCompleted()` or `addCompletedHandler` |
| pinned host buffer (`use_host_ptr`)        | `MTL::ResourceStorageModeShared` (unified memory; no pin needed) |

### Unified memory wins

On Apple silicon, CPU and GPU share physical RAM. `MTL::ResourceStorageModeShared`
buffers are zero-copy accessible from both sides, so the OpenCL "pinned buffer +
map" pattern collapses to direct pointer access via `buf->contents()`. This
simplifies a lot of the host code in `gpu_tree_learner.cpp`.

## Build-time #define injection

OpenCL passes `-D POWER_FEATURE_WORKGROUPS=X -D IGNORE_INDICES ...` as build
options. Metal equivalent: set `MTL::CompileOptions::preprocessorMacros` to an
`NS::Dictionary*` of `NS::String → NS::Number/NS::String`.

```cpp
auto opts = MTL::CompileOptions::alloc()->init();
auto macros = NS::Dictionary::dictionary(...);  // {"POWER_FEATURE_WORKGROUPS": @8, ...}
opts->setPreprocessorMacros(macros);
auto lib = device->newLibrary(src_nsstring, opts, &err);
```

## Phase-by-phase work

### Phase 0 (this doc) — DONE

Inventory, plan, branch.

### Phase 1 — standalone kernel benchmark

Build a single-file harness:

- `tools/metal_bench/main.cpp` + `histogram256.metal`
- Generate synthetic feature/gradient/hessian data (e.g. 1M rows × 256 dense
  features, 256 bins).
- Run both: (a) Metal histogram kernel on M-series GPU; (b) the LightGBM CPU
  histogram path (extracted from `feature_histogram.cpp`).
- Report per-iteration ms.
- **Go/no-go gate**: if Metal < 1.5× CPU, surface to user before Phase 2.

### Phase 2 — full integration

1. Vendor `metal-cpp` into `external_libs/metal-cpp/` (Apple's header bundle).
2. Add CMake `USE_METAL` option + APPLE-only link flags.
3. Port `histogram{16,64,256}.cl` → `.metal`.
4. Write `metal_tree_learner.{h,cpp}` mirroring the OpenCL host wrapper.
5. Wire `device_type=metal` in `tree_learner.cpp`.
6. End-to-end smoke test: train on a small dataset, compare against CPU output
   for numerical agreement (tolerate ULP-level drift in atomics order).

### Phase 3 — tests, bindings, docs

- Unit tests under `tests/` (likely `tests/cpp_tests/`) for kernel correctness
  (Metal vs CPU histogram on fixed input).
- Python wheel build flag (`build-python.sh`): `--metal`.
- README section + `docs/Installation-Guide.rst` macOS-Metal subsection.
- CI: GitHub Actions has `macos-14`/`macos-15` runners with Apple silicon —
  add a job that builds with `USE_METAL=ON` and runs the kernel test.

## Risks / known unknowns

1. **Perf**. M-series GPUs are great for bandwidth-bound work, but gradient
   boosting histogram construction is gather-heavy and atomic-heavy. CPU may
   win or be close. Phase 1 benchmark answers this.
2. **MSL 3.0 floor**. `atomic_float` needs MSL 3.0 (Xcode 14+, macOS 13+,
   Apple silicon). For older Intel Macs we'd need the CAS-loop fallback. We
   target M-series only; declare so in README.
3. **Threadgroup memory size limit**. The 256-bin kernel uses
   `LOCAL_MEM_SIZE = 4 * (sizeof(uint) + 2*sizeof(float)) * 256 = 12 KiB` per
   threadgroup. M-series threadgroup limit is 32 KiB, fine.
4. **Argument indexing**. OpenCL `set_arg(i, x)` maps to MSL kernel parameter
   index `i`. Keep slot assignments identical to OpenCL to avoid churn.
5. **Compile cache**. Boost.Compute caches compiled OpenCL on disk. Metal
   has its own offline shader cache (binary archives, `MTL::BinaryArchive`) —
   wire this up to avoid recompiling on every process start. Phase 2/3.
