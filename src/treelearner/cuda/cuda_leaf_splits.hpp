/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_LEAF_SPLITS_HPP_
#define LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_LEAF_SPLITS_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/bin.h>
#include <LightGBM/tree_split_math.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>

#define NUM_THREADS_PER_BLOCK_LEAF_SPLITS (1024)
#define NUM_DATA_THREAD_ADD_LEAF_SPLITS (6)

namespace LightGBM {

struct CUDALeafSplitsStruct {
 public:
  int leaf_index;
  double sum_of_gradients;
  double sum_of_hessians;
  int64_t sum_of_gradients_hessians;
  data_size_t num_data_in_leaf;
  double gain;
  double leaf_value;
  const data_size_t* data_indices_in_leaf;
  hist_t* hist_in_leaf;
};

class CUDALeafSplits: public NCCLInfo {
 public:
  explicit CUDALeafSplits(const data_size_t num_data);

  ~CUDALeafSplits();

  void Init(const bool use_quantized_grad);

  void InitValues(
    const double lambda_l1, const double lambda_l2,
    const score_t* cuda_gradients, const score_t* cuda_hessians,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf, double* root_sum_gradients, double* root_sum_hessians);

  void InitValues(
    const double lambda_l1, const double lambda_l2,
    const int16_t* cuda_gradients_and_hessians,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf, double* root_sum_gradients, double* root_sum_hessians,
    const score_t* grad_scale, const score_t* hess_scale);

  void InitValues();

  const CUDALeafSplitsStruct* GetCUDAStruct() const { return cuda_struct_.RawDataReadOnly(); }

  CUDALeafSplitsStruct* GetCUDAStructRef() { return cuda_struct_.RawData(); }

  void Resize(const data_size_t num_data);

  // These delegate to the single shared SplitGainMath core (tree_split_math.h)
  // so the CUDA and CPU paths use identical formulas. Names/signatures are kept
  // for the existing device call sites.
  __device__ static double ThresholdL1(double s, double l1) {
    return SplitGainMath::ThresholdL1(s, l1);
  }

  template <bool USE_L1, bool USE_SMOOTHING>
  __device__ static double CalculateSplittedLeafOutput(double sum_gradients,
                                          double sum_hessians, double l1, double l2,
                                          double path_smooth, data_size_t num_data,
                                          double parent_output) {
    return SplitGainMath::CalculateLeafOutput<USE_L1, false, USE_SMOOTHING>(
        sum_gradients, sum_hessians, l1, l2, 0.0, path_smooth, num_data, parent_output);
  }

  template <bool USE_L1>
  __device__ static double GetLeafGainGivenOutput(double sum_gradients,
                                      double sum_hessians, double l1,
                                      double l2, double output) {
    return SplitGainMath::LeafGainGivenOutput<USE_L1>(
        sum_gradients, sum_hessians, l1, l2, output);
  }

  template <bool USE_L1, bool USE_SMOOTHING>
  __device__ static double GetLeafGain(double sum_gradients, double sum_hessians,
                          double l1, double l2,
                          double path_smooth, data_size_t num_data,
                          double parent_output) {
    return SplitGainMath::LeafGain<USE_L1, USE_SMOOTHING>(
        sum_gradients, sum_hessians, l1, l2, path_smooth, num_data, parent_output);
  }

  template <bool USE_L1, bool USE_SMOOTHING>
  __device__ static double GetSplitGains(double sum_left_gradients,
                            double sum_left_hessians,
                            double sum_right_gradients,
                            double sum_right_hessians,
                            double l1, double l2,
                            double path_smooth,
                            data_size_t left_count,
                            data_size_t right_count,
                            double parent_output) {
    return GetLeafGain<USE_L1, USE_SMOOTHING>(sum_left_gradients,
                      sum_left_hessians,
                      l1, l2, path_smooth, left_count, parent_output) +
          GetLeafGain<USE_L1, USE_SMOOTHING>(sum_right_gradients,
                      sum_right_hessians,
                      l1, l2, path_smooth, right_count, parent_output);
  }

 private:
  void LaunchInitValuesEmptyKernel();

  void LaunchInitValuesKernel(
    const double lambda_l1, const double lambda_l2,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf,
    const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf);

  void LaunchInitValuesKernel(
    const double lambda_l1, const double lambda_l2,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf,
    const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf,
    const score_t* grad_scale,
    const score_t* hess_scale);

  // Host memory
  data_size_t num_data_;
  int num_blocks_init_from_gradients_;

  // CUDA memory, held by this object
  CUDAVector<CUDALeafSplitsStruct> cuda_struct_;
  CUDAVector<double> cuda_sum_of_gradients_buffer_;
  CUDAVector<double> cuda_sum_of_hessians_buffer_;
  CUDAVector<int64_t> cuda_sum_of_gradients_hessians_buffer_;

  // CUDA memory, held by other object
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_LEAF_SPLITS_HPP_
