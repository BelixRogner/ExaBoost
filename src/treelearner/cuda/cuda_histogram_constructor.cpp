/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <algorithm>
#include <vector>

namespace LightGBM {

CUDAHistogramConstructor::CUDAHistogramConstructor(
  const Dataset* train_data,
  const int num_leaves,
  const int num_threads,
  const std::vector<uint32_t>& feature_hist_offsets,
  const int min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int gpu_device_id,
  const bool gpu_use_dp,
  const bool use_quantized_grad,
  const int num_grad_quant_bins):
  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  min_data_in_leaf_(min_data_in_leaf),
  min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf),
  gpu_device_id_(gpu_device_id),
  gpu_use_dp_(gpu_use_dp),
  use_quantized_grad_(use_quantized_grad),
  num_grad_quant_bins_(num_grad_quant_bins) {
  num_compact_columns_ = 0;
  max_num_compact_cols_per_partition_ = 0;
  use_compact_view_ = false;
  InitFeatureMetaInfo(train_data, feature_hist_offsets);
  cuda_row_data_.reset(nullptr);
}

CUDAHistogramConstructor::~CUDAHistogramConstructor() {
  gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
}

void CUDAHistogramConstructor::InitFeatureMetaInfo(const Dataset* train_data, const std::vector<uint32_t>& feature_hist_offsets) {
  need_fix_histogram_features_.clear();
  need_fix_histogram_features_num_bin_aligend_.clear();
  feature_num_bins_.clear();
  feature_most_freq_bins_.clear();
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    if (most_freq_bin != 0) {
      need_fix_histogram_features_.emplace_back(feature_index);
      uint32_t num_bin_ref = static_cast<uint32_t>(bin_mapper->num_bin()) - 1;
      uint32_t num_bin_aligned = 1;
      while (num_bin_ref > 0) {
        num_bin_aligned <<= 1;
        num_bin_ref >>= 1;
      }
      need_fix_histogram_features_num_bin_aligend_.emplace_back(num_bin_aligned);
    }
    feature_num_bins_.emplace_back(static_cast<uint32_t>(bin_mapper->num_bin()));
    feature_most_freq_bins_.emplace_back(most_freq_bin);
  }
  feature_hist_offsets_.clear();
  for (size_t i = 0; i < feature_hist_offsets.size(); ++i) {
    feature_hist_offsets_.emplace_back(feature_hist_offsets[i]);
  }
  if (feature_hist_offsets.empty()) {
    num_total_bin_ = 0;
  } else {
    num_total_bin_ = static_cast<int>(feature_hist_offsets.back());
  }
}

void CUDAHistogramConstructor::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  cuda_gradients_ = gradients;
  cuda_hessians_ = hessians;
  cuda_hist_.SetValue(0);
}

void CUDAHistogramConstructor::ZeroHistForLeaf(int /*leaf_index*/) {
  // No-op: BeforeTrain zeroes the entire cuda_hist_ buffer.
}

void CUDAHistogramConstructor::SetFeatureUsedBytree(const std::vector<int8_t>& is_feature_used_bytree) {
  if (cuda_is_feature_used_bytree_.Size() != is_feature_used_bytree.size()) {
    cuda_is_feature_used_bytree_.Resize(is_feature_used_bytree.size());
  }
  CopyFromHostToCUDADevice<int8_t>(cuda_is_feature_used_bytree_.RawData(),
                                   is_feature_used_bytree.data(),
                                   is_feature_used_bytree.size(), __FILE__, __LINE__);
}

void LaunchDiagRead(cudaStream_t stream, const uint8_t* src, uint8_t* dst, int n);
void LaunchTransposeColMajorToRowMajor(
    cudaStream_t stream,
    const uint8_t* staging,
    uint8_t* compact_data,
    const int* partition_for_compact,
    const int* compact_partition_column_offsets,
    data_size_t num_data,
    int total_compact_cols);

// Implemented in cuda_histogram_constructor.cu — does the kernel launch.
void LaunchFillCompactDataKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* slot_src_byte,
  const int* slot_src_stride,
  const size_t* slot_dst_byte,
  const int* slot_dst_stride,
  int total_compact_cols,
  data_size_t num_data);

bool CUDAHistogramConstructor::BuildCompactView(const std::vector<int8_t>& is_feature_used_bytree) {
  use_compact_view_ = false;
  // Gate: only support the standard dense path (uint8 bins, no large-bin partitions, no sparse).
  // This is what our Numerai workload uses; other paths fall back to the full kernel.
  if (cuda_row_data_->is_sparse() || cuda_row_data_->bit_type() != 8 ||
      cuda_row_data_->NumLargeBinPartition() > 0 || use_quantized_grad_) {
    return false;
  }
  if (is_feature_used_bytree.empty()) return false;

  // Read partition info from CUDARowData (host-side).
  const std::vector<int>& src_part_col_offsets = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<uint32_t>& src_col_hist_offsets = cuda_row_data_->host_column_hist_offsets();
  const int num_partitions = static_cast<int>(src_part_col_offsets.size()) - 1;
  if (num_partitions <= 0) return false;

  // Per-partition: compute used columns and their local idx within source partition.
  std::vector<int> compact_part_col_offsets;       // [P+1] cumulative compact cols
  std::vector<int> src_part_stride_h;               // [P] src cols per partition
  std::vector<int> src_local_col_for_compact_h;     // [total_compact_cols]
  std::vector<int> partition_for_compact_h;         // [total_compact_cols]
  std::vector<uint32_t> compact_col_hist_offsets_h; // [total_compact_cols] -- relative-to-partition hist offsets

  compact_part_col_offsets.push_back(0);
  int total_compact = 0;
  for (int p = 0; p < num_partitions; ++p) {
    const int p_start = src_part_col_offsets[p];
    const int p_end = src_part_col_offsets[p + 1];
    const int p_num = p_end - p_start;
    src_part_stride_h.push_back(p_num);
    int used_in_p = 0;
    for (int c = p_start; c < p_end; ++c) {
      // is_feature_used_bytree is indexed by inner_feature_index. For dense single-feature groups
      // (Numerai), inner_feature_index == column_index; the bounds-check guards us if not.
      if (c < static_cast<int>(is_feature_used_bytree.size()) && is_feature_used_bytree[c]) {
        src_local_col_for_compact_h.push_back(c - p_start);
        partition_for_compact_h.push_back(p);
        compact_col_hist_offsets_h.push_back(src_col_hist_offsets[c]);
        ++used_in_p;
        ++total_compact;
      }
    }
    compact_part_col_offsets.push_back(total_compact);
  }

  if (total_compact == 0 || total_compact == src_part_col_offsets.back()) {
    // No win: either no features or all features used. Fall back to full path.
    return false;
  }

  const data_size_t num_data = cuda_row_data_->num_data();

  // Allocate / resize compact buffers.
  const size_t compact_data_bytes = static_cast<size_t>(total_compact) * static_cast<size_t>(num_data);
  if (compact_data_uint8_t_.Size() < compact_data_bytes) {
    compact_data_uint8_t_.Resize(compact_data_bytes);
  }

  // Upload metadata.
  // We need cuda copies of:
  //   compact_part_col_offsets (P+1 ints)
  //   src_part_col_offsets     (P+1 ints) -- already on device, but we have host
  //   src_part_stride_h        (P ints)
  //   src_local_col_for_compact_h (total_compact ints)
  //   partition_for_compact_h     (total_compact ints)
  //   compact_col_hist_offsets_h  (total_compact uint32)

  if (compact_feature_partition_column_index_offsets_.Size() < compact_part_col_offsets.size()) {
    compact_feature_partition_column_index_offsets_.Resize(compact_part_col_offsets.size());
  }
  CopyFromHostToCUDADevice<int>(compact_feature_partition_column_index_offsets_.RawData(),
                                compact_part_col_offsets.data(),
                                compact_part_col_offsets.size(), __FILE__, __LINE__);

  if (compact_column_hist_offsets_.Size() < compact_col_hist_offsets_h.size()) {
    compact_column_hist_offsets_.Resize(compact_col_hist_offsets_h.size());
  }
  CopyFromHostToCUDADevice<uint32_t>(compact_column_hist_offsets_.RawData(),
                                     compact_col_hist_offsets_h.data(),
                                     compact_col_hist_offsets_h.size(), __FILE__, __LINE__);

  // Same partition_hist_offsets as the source (used for global hist write-back position per partition).
  const std::vector<uint32_t>& src_part_hist_offsets = cuda_row_data_->host_partition_hist_offsets();
  if (compact_partition_hist_offsets_.Size() < src_part_hist_offsets.size()) {
    compact_partition_hist_offsets_.Resize(src_part_hist_offsets.size());
  }
  CopyFromHostToCUDADevice<uint32_t>(compact_partition_hist_offsets_.RawData(),
                                     src_part_hist_offsets.data(),
                                     src_part_hist_offsets.size(), __FILE__, __LINE__);

  // When the source is host-pinned (zero-copy), the GPU fill kernel does
  // ~1-byte strided PCIe loads which are extremely inefficient (~1 GB/s effective).
  // For host-mapped sources, do a host-side OpenMP gather into pinned compact_data_host_,
  // then a single bulk cudaMemcpy. Empirically this is ~10× faster than the GPU fill kernel.
  //
  // For GPU-resident sources (small datasets that fit in VRAM), the original
  // GPU fill kernel is fastest.
  // Per-tree fill: when source is host (column-major), do one cudaMemcpy per sampled column.
  // Each is a contiguous num_data byte transfer at ~20 GB/s → ~85 ms for f=0.1 / 6.7M rows.
  auto t_build_start = std::chrono::steady_clock::now();
  if (cuda_row_data_->is_data_host_mapped()) {
    const uint8_t* src_col_major = cuda_row_data_->host_partitioned_data_uint8_t();
    // Compact GPU layout: row-major-in-partition (matches histogram kernel expectation).
    // For each compact col c in partition p, we need to write num_data bytes to
    // compact_data_uint8_t_ at strided positions (compact_stride_p apart per row).
    // To get a contiguous bulk transfer, we instead make compact GPU layout COLUMN-MAJOR
    // within partition: [r] = compact_data[part_offset + col_in_p * num_data + r].
    // The histogram kernel access pattern (`data_ptr[r * num_columns_in_partition + threadIdx.x]`)
    // becomes incorrect for column-major; we emit a separate column-major-aware
    // launcher branch.

    // Lazily allocate scratch buffers only used by this host-mapped path.
    CUDAVector<int> d_partition_for_compact(partition_for_compact_h.size());
    CopyFromHostToCUDADevice<int>(d_partition_for_compact.RawData(),
                                  partition_for_compact_h.data(),
                                  partition_for_compact_h.size(), __FILE__, __LINE__);
    // Multi-stream direct cudaMemcpyAsync per column.
    if (compact_staging_col_major_.Size() < compact_data_bytes) {
      compact_staging_col_major_.Resize(compact_data_bytes);
    }
    static const int N_STREAMS = 4;
    static cudaStream_t copy_streams[N_STREAMS] = {nullptr};
    static cudaEvent_t copy_done[N_STREAMS] = {nullptr};
    static bool streams_init = false;
    if (!streams_init) {
      for (int i = 0; i < N_STREAMS; ++i) {
        cudaStreamCreate(&copy_streams[i]);
        cudaEventCreate(&copy_done[i]);
      }
      streams_init = true;
    }
    for (int compact_col = 0; compact_col < total_compact; ++compact_col) {
      const int p = partition_for_compact_h[compact_col];
      const int src_local = src_local_col_for_compact_h[compact_col];
      const int src_p_start = src_part_col_offsets[p];
      const size_t src_byte_offset = (static_cast<size_t>(src_p_start) + static_cast<size_t>(src_local)) * static_cast<size_t>(num_data);
      const size_t staging_byte_offset = static_cast<size_t>(compact_col) * static_cast<size_t>(num_data);
      cudaMemcpyAsync(compact_staging_col_major_.RawData() + staging_byte_offset,
                      src_col_major + src_byte_offset,
                      num_data,
                      cudaMemcpyHostToDevice,
                      copy_streams[compact_col % N_STREAMS]);
    }
    for (int i = 0; i < N_STREAMS; ++i) {
      cudaEventRecord(copy_done[i], copy_streams[i]);
      cudaStreamWaitEvent(cuda_stream_, copy_done[i], 0);
    }
    // GPU transpose staging (col-major) → compact_data (row-major-in-partition).
    LaunchTransposeColMajorToRowMajor(
        cuda_stream_,
        compact_staging_col_major_.RawData(),
        compact_data_uint8_t_.RawData(),
        d_partition_for_compact.RawData(),
        compact_feature_partition_column_index_offsets_.RawData(),
        num_data,
        total_compact);
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_stream_));
    compact_is_col_major_ = false;  // compact_data is now row-major-in-partition
  } else {
    // Build per-slot src/dst metadata host-side. Each compact slot has a fully
    // computed source byte offset and destination byte offset, so the kernel
    // does no per-thread partition lookups (massive win on this gather pattern).
    std::vector<size_t> slot_src_byte_h(total_compact);
    std::vector<int> slot_src_stride_h(total_compact);
    std::vector<size_t> slot_dst_byte_h(total_compact);
    std::vector<int> slot_dst_stride_h(total_compact);
    for (int s = 0; s < total_compact; ++s) {
      const int p = partition_for_compact_h[s];
      const int p_start = src_part_col_offsets[p];
      const size_t src_p_byte = static_cast<size_t>(p_start) * static_cast<size_t>(num_data);
      const int src_local = src_local_col_for_compact_h[s];
      slot_src_byte_h[s] = src_p_byte + static_cast<size_t>(src_local);
      slot_src_stride_h[s] = src_part_stride_h[p];
      const int compact_part_start = compact_part_col_offsets[p];
      const int compact_stride = compact_part_col_offsets[p + 1] - compact_part_start;
      const int compact_col_in_p = s - compact_part_start;
      const size_t compact_p_byte = static_cast<size_t>(compact_part_start) * static_cast<size_t>(num_data);
      slot_dst_byte_h[s] = compact_p_byte + static_cast<size_t>(compact_col_in_p);
      slot_dst_stride_h[s] = compact_stride;
    }
    if (cuda_slot_src_byte_.Size() < static_cast<size_t>(total_compact)) {
      cuda_slot_src_byte_.Resize(total_compact);
      cuda_slot_src_stride_.Resize(total_compact);
      cuda_slot_dst_byte_.Resize(total_compact);
      cuda_slot_dst_stride_.Resize(total_compact);
    }
    CopyFromHostToCUDADevice<size_t>(cuda_slot_src_byte_.RawData(), slot_src_byte_h.data(), total_compact, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_slot_src_stride_.RawData(), slot_src_stride_h.data(), total_compact, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<size_t>(cuda_slot_dst_byte_.RawData(), slot_dst_byte_h.data(), total_compact, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_slot_dst_stride_.RawData(), slot_dst_stride_h.data(), total_compact, __FILE__, __LINE__);
    LaunchFillCompactDataKernel(
      cuda_stream_,
      cuda_row_data_->GetBin<uint8_t>(),
      compact_data_uint8_t_.RawData(),
      cuda_slot_src_byte_.RawData(),
      cuda_slot_src_stride_.RawData(),
      cuda_slot_dst_byte_.RawData(),
      cuda_slot_dst_stride_.RawData(),
      total_compact,
      num_data);
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_stream_));
    compact_is_col_major_ = false;
  }


  // Compute max compact cols per partition (sets block_dim_x for compact launches).
  int max_compact_per_p = 0;
  for (int p = 0; p < num_partitions; ++p) {
    const int cnt = compact_part_col_offsets[p + 1] - compact_part_col_offsets[p];
    if (cnt > max_compact_per_p) max_compact_per_p = cnt;
  }
  max_num_compact_cols_per_partition_ = max_compact_per_p;

  num_compact_columns_ = total_compact;
  use_compact_view_ = true;
  return true;
}

void CUDAHistogramConstructor::Init(const Dataset* train_data, TrainingShareStates* share_state) {
  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_));
  cuda_hist_.SetValue(0);

  cuda_feature_num_bins_.InitFromHostVector(feature_num_bins_);
  cuda_feature_hist_offsets_.InitFromHostVector(feature_hist_offsets_);
  cuda_feature_most_freq_bins_.InitFromHostVector(feature_most_freq_bins_);

  cuda_row_data_.reset(new CUDARowData(train_data, share_state, gpu_device_id_, gpu_use_dp_));
  cuda_row_data_->Init(train_data, share_state);

  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));

  cuda_need_fix_histogram_features_.InitFromHostVector(need_fix_histogram_features_);
  cuda_need_fix_histogram_features_num_bin_aligned_.InitFromHostVector(need_fix_histogram_features_num_bin_aligend_);

  if (cuda_row_data_->NumLargeBinPartition() > 0) {
    int grid_dim_x = 0, grid_dim_y = 0, block_dim_x = 0, block_dim_y = 0;
    CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_);
    const size_t buffer_size = static_cast<size_t>(grid_dim_y) * static_cast<size_t>(num_total_bin_);
    if (!use_quantized_grad_) {
      if (gpu_use_dp_) {
        // need to double the size of histogram buffer in global memory when using double precision in histogram construction
        cuda_hist_buffer_.Resize(buffer_size * 4);
      } else {
        cuda_hist_buffer_.Resize(buffer_size * 2);
      }
    } else {
      // use only half the size of histogram buffer in global memory when quantized training since each gradient and hessian takes only 2 bytes
      cuda_hist_buffer_.Resize(buffer_size);
    }
  }
  hist_buffer_for_num_bit_change_.Resize(num_total_bin_ * 2);
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* /*cuda_larger_leaf_splits*/,
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  const data_size_t num_data_in_smaller_leaf,
  const data_size_t /*num_data_in_larger_leaf*/,
  const double sum_hessians_in_smaller_leaf,
  const double sum_hessians_in_larger_leaf,
  const uint8_t num_bits_in_histogram_bins) {
if ((global_num_data_in_smaller_leaf <= min_data_in_leaf_ || sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
    (global_num_data_in_larger_leaf <= min_data_in_leaf_ || sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    return;
  }
  LaunchConstructHistogramKernel(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAHistogramConstructor::SubtractHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool use_quantized_grad,
  const uint8_t parent_num_bits_in_histogram_bins,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  LaunchSubtractHistogramKernel(cuda_smaller_leaf_splits, cuda_larger_leaf_splits, use_quantized_grad,
                                parent_num_bits_in_histogram_bins, smaller_num_bits_in_histogram_bins, larger_num_bits_in_histogram_bins);
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
}

void CUDAHistogramConstructor::CalcConstructHistogramKernelDim(
  int* grid_dim_x,
  int* grid_dim_y,
  int* block_dim_x,
  int* block_dim_y,
  const data_size_t num_data_in_smaller_leaf) {
  *block_dim_x = cuda_row_data_->max_num_column_per_partition();
  *block_dim_y = NUM_THREADS_PER_BLOCK / cuda_row_data_->max_num_column_per_partition();
  *grid_dim_x = cuda_row_data_->num_feature_partitions();
  *grid_dim_y = std::max(min_grid_dim_y_,
    ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + (*block_dim_y) - 1) / (*block_dim_y));
}

void CUDAHistogramConstructor::ResetTrainingData(const Dataset* train_data, TrainingShareStates* share_states) {
  num_data_ = train_data->num_data();
  num_features_ = train_data->num_features();
  InitFeatureMetaInfo(train_data, share_states->feature_hist_offsets());

  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_));
  cuda_hist_.SetValue(0);
  cuda_feature_num_bins_.InitFromHostVector(feature_num_bins_);
  cuda_feature_hist_offsets_.InitFromHostVector(feature_hist_offsets_);
  cuda_feature_most_freq_bins_.InitFromHostVector(feature_most_freq_bins_);

  cuda_row_data_.reset(new CUDARowData(train_data, share_states, gpu_device_id_, gpu_use_dp_));
  cuda_row_data_->Init(train_data, share_states);

  cuda_need_fix_histogram_features_.InitFromHostVector(need_fix_histogram_features_);
  cuda_need_fix_histogram_features_num_bin_aligned_.InitFromHostVector(need_fix_histogram_features_num_bin_aligend_);
}

void CUDAHistogramConstructor::ResetConfig(const Config* config) {
  num_threads_ = OMP_NUM_THREADS();
  num_leaves_ = config->num_leaves;
  min_data_in_leaf_ = config->min_data_in_leaf;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_));
  cuda_hist_.SetValue(0);
}

}  // namespace LightGBM

#endif  // USE_CUDA
