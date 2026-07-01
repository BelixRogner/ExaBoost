/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Single source of truth for the per-leaf split-gain / leaf-output numeric
 * math shared by every backend (CPU serial learner, OpenCL, CUDA).
 *
 * These functions used to be duplicated: FeatureHistogram (CPU) and
 * CUDALeafSplits (CUDA) each carried their own copy, which drifted over time
 * (e.g. ThresholdL1 sign handling). Defining them once as __host__ __device__
 * free functions means the CPU and device paths can never disagree again.
 */
#ifndef LIGHTGBM_TREE_SPLIT_MATH_H_
#define LIGHTGBM_TREE_SPLIT_MATH_H_

#include <LightGBM/meta.h>

#include <cmath>

#if defined(__CUDACC__)
#define LGBM_HOSTDEV __host__ __device__
#else
#define LGBM_HOSTDEV
#endif

namespace LightGBM {

namespace SplitGainMath {

// Soft-threshold used for L1 regularization: sign(s) * max(0, |s| - l1).
LGBM_HOSTDEV inline double ThresholdL1(double s, double l1) {
  const double reg_s = fmax(0.0, fabs(s) - l1);
  return s >= 0.0 ? reg_s : -reg_s;
}

// Newton leaf output -g/(h+l2) with optional L1 shrink, max_delta_step cap, and
// path smoothing -- applied in that order (matching the CPU formula). Monotone
// clamping is applied by the caller (it depends on per-leaf constraints).
template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
LGBM_HOSTDEV inline double CalculateLeafOutput(double sum_gradients, double sum_hessians,
                                               double l1, double l2, double max_delta_step,
                                               double path_smooth, data_size_t num_data,
                                               double parent_output) {
  double ret = USE_L1 ? (-ThresholdL1(sum_gradients, l1) / (sum_hessians + l2))
                      : (-sum_gradients / (sum_hessians + l2));
  if (USE_MAX_OUTPUT) {
    if (max_delta_step > 0 && fabs(ret) > max_delta_step) {
      ret = ret >= 0.0 ? max_delta_step : -max_delta_step;
    }
  }
  if (USE_SMOOTHING) {
    ret = ret * (num_data / path_smooth) / (num_data / path_smooth + 1)
        + parent_output / (num_data / path_smooth + 1);
  }
  return ret;
}

// Gain contributed by a leaf given a already-computed output value.
template <bool USE_L1>
LGBM_HOSTDEV inline double LeafGainGivenOutput(double sum_gradients, double sum_hessians,
                                               double l1, double l2, double output) {
  const double g = USE_L1 ? ThresholdL1(sum_gradients, l1) : sum_gradients;
  return -(2.0 * g * output + (sum_hessians + l2) * output * output);
}

// Gain of a leaf (no max_delta_step). With smoothing, gain is measured at the
// smoothed output; without, it collapses to the closed-form g^2/(h+l2).
template <bool USE_L1, bool USE_SMOOTHING>
LGBM_HOSTDEV inline double LeafGain(double sum_gradients, double sum_hessians, double l1,
                                    double l2, double path_smooth, data_size_t num_data,
                                    double parent_output) {
  if (!USE_SMOOTHING) {
    const double g = USE_L1 ? ThresholdL1(sum_gradients, l1) : sum_gradients;
    return (g * g) / (sum_hessians + l2);
  }
  const double output = CalculateLeafOutput<USE_L1, false, USE_SMOOTHING>(
      sum_gradients, sum_hessians, l1, l2, 0.0, path_smooth, num_data, parent_output);
  return LeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
}

}  // namespace SplitGainMath

}  // namespace LightGBM

#endif  // LIGHTGBM_TREE_SPLIT_MATH_H_
