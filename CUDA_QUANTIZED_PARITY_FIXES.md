# CUDA `use_quantized_grad` Parity Fixes

This document describes a cascade of four defects that together left CUDA
quantized gradient training (`device_type=cuda` + `use_quantized_grad=true`)
completely broken on this fork, and the fixes that bring it to parity with the
CPU implementation.

All benchmarks below diff CPU-quantized vs CUDA-quantized predictions on a fixed
synthetic regression dataset (`y = X·w + 0.3·noise`), 20 boosting rounds,
`num_leaves=31`, `gpu_use_dp=true`, `deterministic=true`, single-threaded,
reporting `max|CUDA − CPU|` over all rows.

## Summary

| Fix | Defect | Branch |
|-----|--------|--------|
| #8  | Discretized grad/hess buffer under-allocated 2× (heap overflow) | `cuda/quantized-discretized-buffer-overflow` |
| #9  | Split kernel never refreshes child packed sum | `cuda/quantized-child-leaf-packed-sum` |
| #10 | `CUDASplitInfo::operator=` drops the packed-sum fields | `cuda/quantized-child-leaf-packed-sum` |
| #11 | Histogram-slot collision corrupts small leaves | `cuda/quantized-hist-slot-collision` |

The branches are stacked off `belix/master`:

```
belix/master
  └─ cuda/quantized-discretized-buffer-overflow   (#8 + test)
       └─ cuda/quantized-child-leaf-packed-sum     (#9 + #10 + test)
            └─ cuda/quantized-hist-slot-collision  (#11 + test + this doc)
```

Net result: CUDA quantized training goes from **single-leaf, non-learning trees
with predictions diverging by ~11** to **trees that match CPU's structure with
predictions diverging by ~1 (and bit-identical in the 16-bit-histogram regime)**.
The residual ~1 is the separate cross-feature gain-tie ordering difference,
addressed on `cuda/gain-tie-break`.

---

## Fix #8 — discretized buffer 2× under-allocation

**File:** `src/treelearner/cuda/cuda_gradient_discretizer.hpp`

`CUDAGradientDiscretizer` stores, per data point, an `int16` discretized gradient
and an `int16` discretized hessian. `DiscretizeGradientsKernel` writes them through
a `reinterpret_cast<int16_t*>` view at indices `[2*i]` and `[2*i+1]`, and every
consumer (the histogram constructor and `CUDAInitValuesKernel3`) reads the buffer
back as `int16_t`. That layout needs `2 * sizeof(int16_t) = 4` bytes per point.

The backing buffer is `int8_t` and was sized `num_data * 2` (the original 8-bit
layout) — only 2 bytes per point, half of what the int16 path writes.
`DiscretizeGradientsKernel` therefore overran the buffer by `num_data * 2` bytes,
corrupting the adjacent device allocations, which in practice are the
gradient/hessian dequantization scale buffers (`grad_max_block_buffer_` /
`hess_max_block_buffer_`).

The corrupted dequant scale (≈1.4e-45, the int bit-pattern of the bin count
reinterpreted as a float) made the root leaf's `sum_hessians` a denormal ~0
instead of the true value, so the root failed the
`sum_hessians > min_sum_hessian_in_leaf` validity check and never split. Every
tree collapsed to a single leaf and the model did not learn.

**Fix:** size the buffer `num_data * 4`.

**How it was found:** host-side `CopyFromCUDADeviceToHost` of the scale buffers
bracketing each discretizer kernel — the scales were correct right after
`ReduceBlockMinMaxKernel`, garbage after `DiscretizeGradientsKernel`, pinpointing
the overrun.

**Before → after** (minimal regression, 20 rounds):

| | leaves/tree | max\|CUDA−CPU\| |
|---|---|---|
| Before | 1 (no learning) | 10.99 |
| After  | real splitting trees | 6.81 |

---

## Fix #9 + #10 — child-leaf packed sum not propagated

**Files:** `src/treelearner/cuda/cuda_data_partition.cu`,
`include/LightGBM/cuda/cuda_split_info.hpp`

Two matched defects concerning `sum_of_gradients_hessians` — the `int64` packed
`(gradient << 32) | hessian` leaf total that the **discretized** best-split finder
uses to derive `cnt_factor = num_data / sum_hessians` and the
`left = total − right` split sums.

- **#10** (`cuda_split_info.hpp`): `CUDASplitInfo::operator=` copied
  `left_/right_sum_gradients`, `_sum_hessians`, `_count`, `_gain`, `_value` but
  **not** `left_/right_sum_of_gradients_hessians`. `FindBestFromAllSplits` assigns
  the winning split through this operator, so the packed sums were dropped (read
  back as 0).
- **#9** (`cuda_data_partition.cu`, `SplitTreeStructureKernel`): the child leaf
  splits had their `double` `sum_of_gradients` / `sum_of_hessians` refreshed on
  split but never their packed `sum_of_gradients_hessians`, so each child inherited
  the parent's total.

Combined effect: at the first sub-split the finder used the child's real per-bin
histogram but the parent's stale (or zeroed) total, scoring phantom
parent-remainder splits that partitioned to a 0-data leaf — growth halted at 2–4
leaves while CPU grew the full 31.

**How it was found:** a host trace after `ApplySplit` printing the finder's
predicted left/right counts vs the actual partition. A 429-data leaf showed
`pred left_count=245 / right=184` from a parent total of 1000 (`cnt_factor =
245/571 = 429/1000`), but the threshold put all 429 on one side.

**Before → after** (leaf counts over 20 rounds, CPU vs CUDA):

| n | min_data | CPU leaves | CUDA leaves | max\|CUDA−CPU\| |
|---|---|---|---|---|
| 1000 | 300 | 54 | 54 | **0 (bit-identical)** |
| 1000 | 200 | 76 | 75 | small |
| 2000 | 300 | 100 | 106 | small |

This is a **structural** fix — trees now grow to match CPU. Predictions were
exact in some configs but still diverged in others, due to fix #11 below.

---

## Fix #11 — histogram-slot collision corrupts small leaves

**File:** `src/treelearner/cuda/cuda_data_partition.cu`
(`SplitTreeStructureKernel`)

Histogram slots in `cuda_hist_` are laid out at a `2 * num_total_bin` stride per
leaf — the buffer is sized `num_total_bin * 2 * num_leaves`, and the
right-is-smaller branch always assigns
`cuda_hist + 2 * right_leaf_index * num_total_bin`. But the left-is-smaller branch
used a **1× stride** for the discretized path:

```cpp
cuda_hist_pool[left_leaf_index] = USE_GRAD_DISCRETIZED ?
    cuda_hist + right_leaf_index * num_total_bin :       // 1x  <-- bug
    cuda_hist + 2 * right_leaf_index * num_total_bin;    // 2x
```

A 1×-strided child could be handed a slot already owned by a live 2×-strided leaf.
Since `ZeroHistForLeaf` is a no-op (the whole `cuda_hist_` buffer is zeroed once
per tree in `BeforeTrain`), the child's `ConstructHistogramForLeaf` then
`atomicAdd`ed its data on top of the other leaf's histogram.

The polluted histogram gave the finder phantom splits whose threshold the data
partition could not reproduce (it routed all data to one side → 0-data leaves),
and leaves ended with near-zero `sum_hessian` whose outputs exploded — compounding
to 1e12–1e20 over 20 rounds at small `min_data_in_leaf`.

**How it was found:** dumping each smaller leaf's constructed histogram total and
its slot offset. For `n` points × `F` features, a clean leaf's histogram hessian
sums to `n * F * 4`. A 45-point leaf summed to `3392 = (61 + 45) * 32` and shared
`slot_off=3264` with a 61-point leaf — a direct slot collision.

**Fix:** use the same 2× stride in both branches. Slots are then distinct and the
allocation still fits exactly (`2 * num_leaves * num_total_bin`).

**Before → after** (max\|CUDA−CPU\|, 20 rounds, with #8+#9+#10 in place):

| n | min_data | Before | After | CUDA / CPU leaves |
|---|---|---|---|---|
| 1000 | 20  | ~6e12  | **0.997** | 620 / 620 |
| 2000 | 20  | ~1e20  | **1.525** | 620 / 620 |
| 1000 | 200 | exploded | **0** | 76 / 76 |
| 2000 | 200 | exploded | **0** | 152 / 152 |

A disproven hypothesis (recorded so it isn't retried): "it's the 8-bit histogram
path / bit-width." Forcing a 16-bit minimum made divergence *worse*
(6e12 → 4.7e20). The bug was the slot stride, not the bit-width.

---

## Testing

Regression tests live in `tests/python_package_test/test_dual.py`
(run with `TASK=cuda`):

- `test_cuda_quantized_training_produces_splits` — guards #8 (CUDA quantized must
  produce multi-leaf trees, not collapse to a single leaf).
- `test_cuda_quantized_tree_structure_matches_cpu` — guards #9/#10 (CUDA grows to
  within 20% of CPU's leaf count in the 16-bit regime).
- `test_cuda_quantized_deep_trees_track_cpu` — guards #11 (deep trees at
  `min_data_in_leaf=20` match CPU's leaf count and predictions stay bounded,
  `max|CUDA−CPU| < 5`).

## Remaining divergence

After all four fixes the residual `max|CUDA−CPU|` of ~1–1.5 at small
`min_data_in_leaf` is the cross-feature gain-tie ordering difference (CUDA's
best-split reduction breaks exact-gain ties differently from CPU's left-to-right
scan). That is a separate, objective-independent issue addressed on the
`cuda/gain-tie-break` branch; stacking it on top brings the remaining difference
to FP-epsilon.
