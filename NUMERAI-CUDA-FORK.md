# LightGBM CUDA fork

This fork patches LightGBM v4.6.0's CUDA backend (`device=cuda`) so it works
correctly and quickly on **wide, large datasets that don't fit twice in
GPU memory** — concretely tabular workloads on the order of 6.7 M rows ×
2748 features (~17 GB just for the binned matrix on GPU). On a 32 GB
RTX 5090 the upstream backend cannot construct the dataset (it tries to
keep both a row-major and a column-major copy, ~34 GB total) and even when
it can, three latent bugs corrupt training silently or degrade throughput.

## What this fork adds

Three are bug fixes targeted at upstream — small, isolated, with reproducers.
The fourth is a feature substantial enough to deserve a wider design review
before it lands upstream; for now it lives only here.

| Patch | Branch | Lines | PR target |
|---|---|---:|---|
| Fix int32 overflow in dense histogram address arithmetic | [`fix-cuda-hist-int32-overflow`](https://github.com/BelixRogner/LightGBM/tree/fix-cuda-hist-int32-overflow) | 4 | upstream |
| Fix half-size allocation of discretized gradient buffer | [`fix-cuda-discretizer-buffer-size`](https://github.com/BelixRogner/LightGBM/tree/fix-cuda-discretizer-buffer-size) | 1 | upstream |
| Destroy per-tree CUDA stream in `CUDATree::ToHost` | [`fix-cuda-tree-stream-leak`](https://github.com/BelixRogner/LightGBM/tree/fix-cuda-tree-stream-leak) | 16 | upstream |
| Per-tree feature-fraction compact view + GPU-resident bin matrix with host-pinned fallback | [`numerai-cuda-fast`](https://github.com/BelixRogner/LightGBM/tree/numerai-cuda-fast) | ~1000 | this fork only |

The rolling `numerai-cuda-fast` branch contains all four.

### Bug 1 — int32 overflow in histogram address arithmetic

`CUDAConstructHistogramDenseKernel` and the discretized-grad variant
compute byte offsets into the global bin matrix as
`partition_column_start * num_data` and
`data_index * num_columns_in_partition` in `int32`. With wide datasets
these products exceed `2^31` and silently wrap, causing the kernel to
read garbage memory — corrupted histograms, occasionally
`cudaErrorIllegalAddress`.

**Trigger**: any `device='cuda'` training where
`max_partition_columns * num_data > 2^31`. With LightGBM's default
6 partitions and ~3 000 features, this hits at roughly 750 K rows.

**Fix**: cast both factors to `size_t` before the multiply (4 lines).
`compute-sanitizer`-confirmed.

### Bug 2 — discretizer buffer half-sized

`CUDAGradientDiscretizer::Init` resizes a `CUDAVector<int8_t>` to
`num_data * 2` *elements*. The `DiscretizeGradientsKernel` writes a *pair*
of `int16` values per data row — i.e. it needs `num_data * 4` *bytes*.
The buffer is half the required size; the kernel writes past the end for
the upper half of the data on every quantized-grad run.

**Trigger**: any `use_quantized_grad=True` training of any size; just
rarely caught because `compute-sanitizer` is needed to see the
out-of-bounds write on most GPUs.

**Fix**: `Resize(num_data * 4)` (one line).

### Bug 3 — per-tree CUDA stream leak

Each `CUDATree` creates a `cudaStream_t` in its constructor and only
destroys it when the booster shuts down. The stream is only used by
`SplitKernel`/`SplitCategoricalKernel` during construction and is dead
weight afterward, but the booster's `models_` list keeps the `CUDATree`
(and its stream) alive until shutdown. CUDA driver scheduling overhead
grows linearly with the number of live streams.

**Empirical measurement** (RTX 5090, 6.7 M-row dense tabular dataset,
`num_leaves=8192`, `colsample_bytree=0.1`, `learning_rate=0.001`,
non-quantized):

| Iteration | Without fix | With fix |
|--:|--:|--:|
| 1 000 | 11.8 TPS | 13.4 TPS |
| 5 000 | 7.7 TPS | 13.4 TPS |
| 9 000 | 5.4 TPS | 13.4 TPS |
| 30 000 | (continues degrading) | 13.4 TPS |

Without the fix, throughput halves over the first 9 000 trees and OOMs
around iter 6 000 due to combined stream + per-tree-buffer accumulation;
with the fix, throughput is constant.

**Fix**: destroy the stream in `ToHost()`; null-guard the destructor
(~16 lines).

### Feature 4 — per-tree feature-fraction compact view

When `colsample_bytree < 1` (typical: 0.1 for highly-overparameterized
models on noisy data), the upstream CUDA backend still iterates *all*
feature columns inside the histogram kernel. This fork adds two per-tree
compact buffers, both produced from the on-GPU row-major bin matrix:

* A **row-major-in-partition compact bin matrix** (consumed by the
  existing histogram kernel — no kernel changes needed beyond pointing
  it at the compact data).
* A **column-major compact buffer** (consumed by the partition-split
  kernel, which needs per-column lookups).

Both are filled by slot-keyed kernels that touch only the sampled
features. At `f=0.1`, this is ~10× less histogram work and ~10× less
partition-split work per tree.

The feature also includes:

* `CUDARowData` graceful host-pinned fallback for >32 GB GPU bin
  matrices (zero-copy, slow per-tree gather, but won't OOM at construct
  time).
* `CUDAColumnData` skipping the redundant 17 GB per-column allocation
  when total > 8 GB; partition kernels read from the per-tree compact
  column buffer via `SetCompactColumnView`.
* `CUDATree::ToHost` aggressive free of all per-tree GPU buffers except
  `cuda_leaf_value_` (still read by `AddPredictionToScore`), shrunk to
  actual `num_leaves_`. Saves ~63 KB per tree → ~3.8 GB at 60 k trees.
* `AsConstantTree` lazy realloc / null guards to cooperate with the
  above.

## Throughput summary

Same 6.7 M × 2 748 dense tabular dataset; non-quantized; `max_bin=5`,
`colsample_bytree=0.1`, `num_leaves=8192`; RTX 5090 (32 GB, sm_120):

| Configuration | TPS | Per-iter | Notes |
|---|--:|--:|---|
| OpenCL `device=gpu` (upstream) | 5.23 | 191 ms | reference |
| CUDA `device=cuda` (upstream `v4.6.0`) | n/a | n/a | OOMs at dataset construct (17 GB col matrix) |
| CUDA + 3 bug fixes (no compact view) | n/a | n/a | OOMs at iter ~6 k due to stream + per-tree buffer accumulation |
| CUDA + 3 bug fixes + stream destroy | 11.9 | 84 ms | works, but no feature-fraction speedup |
| CUDA + 3 bug fixes + compact view (this fork) | **13.4** | **75 ms** | stable to 60 k trees |

Predictions are bit-equivalent to OpenCL within `float32` epsilon:
`max abs diff = 5.96e-8`, `pearson = 1.000000`.

## Build

```bash
git clone https://github.com/BelixRogner/LightGBM.git
cd LightGBM
git checkout numerai-cuda-fast
git submodule update --init --recursive
mkdir build && cd build
# Adjust CMAKE_CUDA_ARCHITECTURES for your GPU. RTX 5090 = 120, RTX 4090 = 89.
cmake -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="89-real;120-real;120-virtual" ..
cmake --build . --target _lightgbm -j 8
```

Then install the Python package against this build per upstream's
`python-package/build-python.sh --precompile` instructions.

## Why a fork

Three of the four patches are submitted as separate single-commit
upstream PRs. Once those merge, this fork collapses to just the
per-tree compact-view feature, which is a larger architectural change
that benefits from operating-cost real data before going upstream. We
rebase onto upstream periodically.
