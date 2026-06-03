/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

// =====================================================================
// Compaction kernel: copies sampled (used) columns from the partitioned
// row-major bin matrix into a compact partitioned buffer.
//
// Source layout (per partition p):
//   src_data[partition_byte_offset[p] + row * src_stride[p] + col_in_partition]
// where partition_byte_offset[p] = src_partition_column_start[p] * num_data
//       src_stride[p]            = src_num_columns_in_partition[p]
//
// Compact layout (per partition p, packed contiguous over USED columns):
//   compact_data[compact_byte_offset[p] + row * compact_stride[p] + i_in_partition]
// where compact_byte_offset[p] = compact_partition_column_start[p] * num_data
//       compact_stride[p]      = num_used_in_partition[p]
//
// One thread copies one (row, compact_col) pair. Grid is sized as
// (ceil(total_compact_cols / TX), ceil(num_data / TY)).
// =====================================================================
// Per-slot precomputed metadata fill kernel. Writes row-major-in-partition
// output. Block is 32 slots × 32 rows. Each thread copies bytes down a column
// to keep grid_y under CUDA's 65535 limit on large datasets.
__global__ void CUDAFillCompactDataKernel(
  const uint8_t* __restrict__ src_data,
  uint8_t* __restrict__ compact_data,
  const size_t* __restrict__ slot_src_byte,
  const int* __restrict__ slot_src_stride,
  const size_t* __restrict__ slot_dst_byte,
  const int* __restrict__ slot_dst_stride,
  const int total_compact_cols,
  const data_size_t num_data) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= total_compact_cols) return;
  const size_t src_byte = slot_src_byte[slot];
  const size_t src_stride = static_cast<size_t>(slot_src_stride[slot]);
  const size_t dst_byte = slot_dst_byte[slot];
  const size_t dst_stride = static_cast<size_t>(slot_dst_stride[slot]);
  const data_size_t row_stride = static_cast<data_size_t>(gridDim.y) * static_cast<data_size_t>(blockDim.y);
  for (data_size_t row = blockIdx.y * blockDim.y + threadIdx.y; row < num_data; row += row_stride) {
    compact_data[dst_byte + static_cast<size_t>(row) * dst_stride] =
        src_data[src_byte + static_cast<size_t>(row) * src_stride];
  }
}

// Transpose row-major-in-partition source data into column-major compact buffer.
// Source: cuda_data_uint8_t_ (row-major-in-partition).
// Dest: compact_col_buf, layout = compact_col_buf[slot * num_data + row].
//
// Args:
//   src_data:                       cuda_data_uint8_t_
//   compact_col_buf:                destination
//   src_partition_column_offsets:   [P+1] cumulative source col counts (partition byte offset = col_offset * num_data)
//   src_partition_stride:           [P]   columns per source partition (byte stride per row)
//   slot_for_col:                   [num_total_cols] -> compact slot, or -1 if not in sample
//   num_data, num_partitions
// Per-slot precomputed source-frame metadata: for compact slot s,
//   slot_p_byte[s] = partition_byte_offset for that slot's source column
//   slot_p_stride[s] = partition row stride
//   slot_col_in_p[s] = column index within partition
__global__ void CUDARowToColCompactKernel(
    const uint8_t* __restrict__ src_data,
    uint8_t* __restrict__ compact_col_buf,
    const size_t* __restrict__ slot_p_byte,
    const int* __restrict__ slot_p_stride,
    const int* __restrict__ slot_col_in_p,
    const int num_compact_cols,
    const data_size_t num_data) {
  // Block is (32 rows, 32 slots). Each block tile-transposes a 32×32 chunk via
  // shared memory: coalesced reads of 32 contiguous slots × 1 row, coalesced
  // writes of 1 slot × 32 contiguous rows. To stay under CUDA's 65535 grid_y
  // limit on large datasets (>~2M rows), grid_y is capped and the block strides
  // down its column.
  __shared__ uint8_t tile[32][33];

  const int slot_block = blockIdx.x * 32;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const data_size_t y_stride = static_cast<data_size_t>(gridDim.y) * 32;

  for (data_size_t row_block = static_cast<data_size_t>(blockIdx.y) * 32;
       row_block < num_data; row_block += y_stride) {
    // Phase 1: load src[row_block+ty, slot_block+tx] -> tile[ty][tx].
    {
      const int slot = slot_block + tx;
      const data_size_t row = row_block + ty;
      uint8_t val = 0;
      if (slot < num_compact_cols && row < num_data) {
        const size_t base = slot_p_byte[slot]
                          + static_cast<size_t>(row) * static_cast<size_t>(slot_p_stride[slot])
                          + static_cast<size_t>(slot_col_in_p[slot]);
        val = src_data[base];
      }
      tile[ty][tx] = val;
    }
    __syncthreads();

    // Phase 2: write compact[slot_block+ty, row_block+tx] = tile[tx][ty].
    {
      const int slot = slot_block + ty;
      const data_size_t row = row_block + tx;
      if (slot < num_compact_cols && row < num_data) {
        compact_col_buf[static_cast<size_t>(slot) * static_cast<size_t>(num_data) + static_cast<size_t>(row)] = tile[tx][ty];
      }
    }
    __syncthreads();
  }
}

void LaunchRowToColCompactKernel(
    cudaStream_t stream,
    const uint8_t* src_data,
    uint8_t* compact_col_buf,
    const size_t* slot_p_byte,
    const int* slot_p_stride,
    const int* slot_col_in_p,
    int num_compact_cols,
    data_size_t num_data) {
  const int TX = 32;
  const int TY = 32;
  // Cap grid_y at 32k to stay under CUDA's 65535 limit; kernel strides over rows.
  int grid_y = (num_data + TY - 1) / TY;
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((num_compact_cols + TX - 1) / TX, grid_y);
  CUDARowToColCompactKernel<<<grid_dim, block_dim, 0, stream>>>(
      src_data, compact_col_buf, slot_p_byte, slot_p_stride, slot_col_in_p,
      num_compact_cols, num_data);
}

// Transpose column-major staging into row-major-in-partition compact_data.
// staging layout: staging[c * num_data + r] for compact col c (in partition order).
// dst layout:     compact[part_offset[p] + r * stride[p] + c_in_p].
//
// One thread = one (compact_col, row) cell. Threads in same warp have consecutive
// compact_col → consecutive byte writes in dst (within partition; coalesced).
// Reads from staging are stride num_data per col → 32 different cache lines per warp.
// But staging is GPU-resident (HBM, 1.79 TB/s bandwidth) so this is fine.
__global__ void CUDATransposeColMajorToRowMajorKernel(
    const uint8_t* __restrict__ staging,        // col-major: [c * num_data + r]
    uint8_t* __restrict__ compact_data,         // row-major-in-partition
    const int* __restrict__ partition_for_compact,
    const int* __restrict__ compact_partition_column_offsets,
    const data_size_t num_data,
    const int total_compact_cols) {
  const int compact_col = blockIdx.x * blockDim.x + threadIdx.x;
  const data_size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (compact_col >= total_compact_cols) return;
  if (row >= num_data) return;

  const int p = partition_for_compact[compact_col];
  const int p_compact_start = compact_partition_column_offsets[p];
  const int compact_col_in_p = compact_col - p_compact_start;
  const int compact_stride_p = compact_partition_column_offsets[p + 1] - p_compact_start;
  const size_t compact_part_byte_offset = static_cast<size_t>(p_compact_start) * static_cast<size_t>(num_data);

  const uint8_t val = staging[static_cast<size_t>(compact_col) * static_cast<size_t>(num_data) + row];
  compact_data[compact_part_byte_offset + static_cast<size_t>(row) * static_cast<size_t>(compact_stride_p) + compact_col_in_p] = val;
}

void LaunchTransposeColMajorToRowMajor(
    cudaStream_t stream,
    const uint8_t* staging,
    uint8_t* compact_data,
    const int* partition_for_compact,
    const int* compact_partition_column_offsets,
    data_size_t num_data,
    int total_compact_cols) {
  const int TX = 8;
  const int TY = 128;  // grid_y under 65535 for 8M rows
  dim3 block_dim(TX, TY);
  dim3 grid_dim((total_compact_cols + TX - 1) / TX, (num_data + TY - 1) / TY);
  CUDATransposeColMajorToRowMajorKernel<<<grid_dim, block_dim, 0, stream>>>(
      staging, compact_data, partition_for_compact, compact_partition_column_offsets,
      num_data, total_compact_cols);
}

// Diagnostic kernel: read N bytes from a (possibly host-mapped) source pointer.
__global__ void DiagReadKernel(const uint8_t* __restrict__ src, uint8_t* dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
void LaunchDiagRead(cudaStream_t stream, const uint8_t* src, uint8_t* dst, int n) {
  DiagReadKernel<<<(n + 31) / 32, 32, 0, stream>>>(src, dst, n);
}

// Host wrapper called from cuda_histogram_constructor.cpp.
void LaunchFillCompactDataKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* slot_src_byte,
  const int* slot_src_stride,
  const size_t* slot_dst_byte,
  const int* slot_dst_stride,
  int total_compact_cols,
  data_size_t num_data) {
  const int TX = 32;
  const int TY = 32;
  // Cap grid_y at 32k so we stay well under CUDA's 65535 limit; the kernel
  // strides each thread down the column to cover all rows.
  int grid_y = (num_data + TY - 1) / TY;
  if (grid_y > 32768) grid_y = 32768;
  dim3 block_dim(TX, TY);
  dim3 grid_dim((total_compact_cols + TX - 1) / TX, grid_y);
  CUDAFillCompactDataKernel<<<grid_dim, block_dim, 0, stream>>>(
    src_data,
    compact_data,
    slot_src_byte,
    slot_src_stride,
    slot_dst_byte,
    slot_dst_stride,
    total_compact_cols,
    num_data);
}

// Column-major-in-partition variant of the dense histogram kernel.
// Used by the CompactView host-mapped path on large datasets, where compact_data
// is laid out column-major-per-partition for cheap per-column cudaMemcpy fill.
template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramDenseColMajorKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  // data is column-major-in-partition: column c at offset partition_column_start + threadIdx.x
  // starts at byte (partition_column_start + threadIdx.x) * num_data.
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    // Column-major: this thread's column starts at data_ptr + threadIdx.x * num_data.
    const BIN_TYPE* col_ptr = data_ptr + static_cast<size_t>(threadIdx.x) * static_cast<size_t>(num_data);
    for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y); inner_data_index < block_num_data; inner_data_index += blockDim.y) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(col_ptr[data_index]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const int8_t* is_feature_used_bytree,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  // Skip features that are not in the per-tree feature_fraction sample.
  // is_feature_used_bytree may be nullptr (no sampling); treat as all-used.
  const bool feat_used = (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) &&
      (is_feature_used_bytree == nullptr || is_feature_used_bytree[column_index]);
  if (feat_used) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y); inner_data_index < block_num_data; inner_data_index += blockDim.y) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[static_cast<size_t>(data_index) * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + static_cast<size_t>(blockIdx.x) * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename HIST_TYPE>
__global__ void CUDAConstructHistogramDenseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data,
  HIST_TYPE* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  HIST_TYPE* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[static_cast<size_t>(data_index) * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename HIST_TYPE, typename DATA_PTR_TYPE>
__global__ void CUDAConstructHistogramSparseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  HIST_TYPE* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + static_cast<size_t>(blockIdx.x) * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  HIST_TYPE* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + partition_column_start * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    int32_t* shared_hist_ptr = shared_hist_packed + (column_hist_offsets[column_index]);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[data_index * num_columns_in_partition + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_ptr + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ int16_t shared_hist[SHARED_HIST_SIZE];
  int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_packed + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data,
  int32_t* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + partition_column_start * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  int32_t* shared_hist_packed = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    int32_t* shared_hist_ptr = shared_hist_packed + (column_hist_offsets[column_index]);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[data_index * num_columns_in_partition + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_ptr + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, int SHARED_HIST_SIZE, bool USE_16BIT_HIST>
__global__ void CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int32_t* cuda_gradients_and_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  int32_t* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + blockIdx.x * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start);
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  int32_t* shared_hist_packed = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist_packed[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const int32_t grad_and_hess = cuda_gradients_and_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      int32_t* pos_ptr = shared_hist_packed + bin;
      atomicAdd_block(pos_ptr, grad_and_hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  if (USE_16BIT_HIST) {
    int32_t* feature_histogram_ptr = reinterpret_cast<int32_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      atomicAdd_system(feature_histogram_ptr + i, packed_grad_hess);
    }
  } else {
    atomic_add_long_t* feature_histogram_ptr = reinterpret_cast<atomic_add_long_t*>(smaller_leaf_splits->hist_in_leaf) + partition_hist_start;
    for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
      const int32_t packed_grad_hess = shared_hist_packed[i];
      const int64_t packed_grad_hess_int64 = (static_cast<int64_t>(static_cast<int16_t>(packed_grad_hess >> 16)) << 32) | (static_cast<int64_t>(packed_grad_hess & 0x0000ffff));
      atomicAdd_system(feature_histogram_ptr + i, (atomic_add_long_t)(packed_grad_hess_int64));
    }
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->shared_hist_size() == DP_SHARED_HIST_SIZE && gpu_use_dp_) {
    LaunchConstructHistogramKernelInner<double, DP_SHARED_HIST_SIZE>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->shared_hist_size() == SP_SHARED_HIST_SIZE && !gpu_use_dp_) {
    LaunchConstructHistogramKernelInner<float, SP_SHARED_HIST_SIZE>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    Log::Fatal("Unknown shared histogram size %d", cuda_row_data_->shared_hist_size());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->bit_type() == 8) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint8_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->bit_type() == 16) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->bit_type() == 32) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint32_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    Log::Fatal("Unknown bit_type = %d", cuda_row_data_->bit_type());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->row_ptr_bit_type() == 16) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint32_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint64_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    if (!cuda_row_data_->is_sparse()) {
      LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
    } else {
      Log::Fatal("Unknown row_ptr_bit_type = %d", cuda_row_data_->row_ptr_bit_type());
    }
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner1(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  if (cuda_row_data_->NumLargeBinPartition() == 0) {
    LaunchConstructHistogramKernelInner2<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, PTR_TYPE, false>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  } else {
    LaunchConstructHistogramKernelInner2<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, PTR_TYPE, true>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE, bool USE_GLOBAL_MEM_BUFFER>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner2(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf,
  const uint8_t num_bits_in_histogram_bins) {
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_in_smaller_leaf);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  if (use_quantized_grad_) {
    if (USE_GLOBAL_MEM_BUFFER) {
      if (cuda_row_data_->is_sparse()) {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        }
      } else {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory<BIN_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_,
            reinterpret_cast<int32_t*>(cuda_hist_buffer_.RawData()));
        }
      }
    } else {
      if (cuda_row_data_->is_sparse()) {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramSparseKernel<BIN_TYPE, PTR_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->GetRowPtr<PTR_TYPE>(),
            cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            num_data_);
        }
      } else {
        if (num_bits_in_histogram_bins <= 16) {
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, true><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_);
        } else {
          CUDAConstructDiscretizedHistogramDenseKernel<BIN_TYPE, SHARED_HIST_SIZE, false><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            reinterpret_cast<const int32_t*>(cuda_gradients_),
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            num_data_);
        }
      }
    }
  } else {
    if (!USE_GLOBAL_MEM_BUFFER) {
      if (cuda_row_data_->is_sparse()) {
        CUDAConstructHistogramSparseKernel<BIN_TYPE, PTR_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->GetBin<BIN_TYPE>(),
          cuda_row_data_->GetRowPtr<PTR_TYPE>(),
          cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          num_data_);
      } else {
        // ====== COMPACT VIEW PATH (feature_fraction sampling honored on GPU) ======
        if (use_compact_view_) {
          // DEBUG: use SOURCE pointers but with my computed compact dims to isolate bug
          const int compact_block_x = max_num_compact_cols_per_partition_;
          const int compact_block_y = NUM_THREADS_PER_BLOCK / std::max(1, compact_block_x);
          const int compact_grid_y = std::max(min_grid_dim_y_,
              ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + std::max(1, compact_block_y) - 1) / std::max(1, compact_block_y));
          dim3 compact_grid_dim(grid_dim_x, compact_grid_y);
          dim3 compact_block_dim(compact_block_x, std::max(1, compact_block_y));
          // After BuildCompactView swap, compact_data_uint8_t_ is whichever buffer is now active.
          // (When use_compact_view_ true and host-mapped path is used, BuildCompactView swaps
          // active_buffer_is_alt_; the "active" buffer for histograms is the OPPOSITE of
          // active_buffer_is_alt_ after the swap, since BuildCompactView fills the "alt"
          // and then flips active_buffer_is_alt_.)
          uint8_t* active_data = compact_data_uint8_t_.RawData();
          if (compact_is_col_major_) {
            CUDAConstructHistogramDenseColMajorKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<compact_grid_dim, compact_block_dim, 0, cuda_stream_>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              reinterpret_cast<const BIN_TYPE*>(active_data),
              compact_column_hist_offsets_.RawData(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              compact_feature_partition_column_index_offsets_.RawData(),
              num_data_);
          } else {
            CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<compact_grid_dim, compact_block_dim, 0, cuda_stream_>>>(
              cuda_smaller_leaf_splits,
              cuda_gradients_, cuda_hessians_,
              reinterpret_cast<const BIN_TYPE*>(active_data),
              compact_column_hist_offsets_.RawData(),
              cuda_row_data_->cuda_partition_hist_offsets(),
              compact_feature_partition_column_index_offsets_.RawData(),
              nullptr,
              num_data_);
          }
        } else {
          CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
            cuda_smaller_leaf_splits,
            cuda_gradients_, cuda_hessians_,
            cuda_row_data_->GetBin<BIN_TYPE>(),
            cuda_row_data_->cuda_column_hist_offsets(),
            cuda_row_data_->cuda_partition_hist_offsets(),
            cuda_row_data_->cuda_feature_partition_column_index_offsets(),
            cuda_is_feature_used_bytree_.Size() > 0 ? cuda_is_feature_used_bytree_.RawData() : nullptr,
            num_data_);
        }
      }
    } else {
      if (cuda_row_data_->is_sparse()) {
        CUDAConstructHistogramSparseKernel_GlobalMemory<BIN_TYPE, HIST_TYPE, PTR_TYPE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->GetBin<BIN_TYPE>(),
          cuda_row_data_->GetRowPtr<PTR_TYPE>(),
          cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          num_data_,
          reinterpret_cast<HIST_TYPE*>(cuda_hist_buffer_.RawData()));
      } else {
        CUDAConstructHistogramDenseKernel_GlobalMemory<BIN_TYPE, HIST_TYPE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
          cuda_smaller_leaf_splits,
          cuda_gradients_, cuda_hessians_,
          cuda_row_data_->GetBin<BIN_TYPE>(),
          cuda_row_data_->cuda_column_hist_offsets(),
          cuda_row_data_->cuda_partition_hist_offsets(),
          cuda_row_data_->cuda_feature_partition_column_index_offsets(),
          num_data_,
          reinterpret_cast<HIST_TYPE*>(cuda_hist_buffer_.RawData()));
      }
    }
  }
}

__global__ void SubtractHistogramKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index >= 0) {
    const hist_t* smaller_leaf_hist = cuda_smaller_leaf_splits->hist_in_leaf;
    hist_t* larger_leaf_hist = cuda_larger_leaf_splits->hist_in_leaf;
    if (global_thread_index < 2 * num_total_bin) {
      larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
    }
  }
}

__global__ void FixHistogramKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ hist_t shared_mem_buffer[WARPSIZE];
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  const double leaf_sum_gradients = cuda_smaller_leaf_splits->sum_of_gradients;
  const double leaf_sum_hessians = cuda_smaller_leaf_splits->sum_of_hessians;
  hist_t* feature_hist = cuda_smaller_leaf_splits->hist_in_leaf + feature_hist_offset * 2;
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t num_bin = cuda_feature_num_bins[feature_index];
  const uint32_t hist_pos = threadIdx_x << 1;
  const hist_t bin_gradient = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos] : 0.0f;
  const hist_t bin_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos + 1] : 0.0f;
  const hist_t sum_gradient = ShuffleReduceSum<hist_t>(bin_gradient, shared_mem_buffer, num_bin_aligned);
  const hist_t sum_hessian = ShuffleReduceSum<hist_t>(bin_hessian, shared_mem_buffer, num_bin_aligned);
  if (threadIdx_x == 0) {
    feature_hist[most_freq_bin << 1] = leaf_sum_gradients - sum_gradient;
    feature_hist[(most_freq_bin << 1) + 1] = leaf_sum_hessians - sum_hessian;
  }
}

template <bool SMALLER_USE_16BIT_HIST, bool LARGER_USE_16BIT_HIST, bool PARENT_USE_16BIT_HIST>
__global__ void SubtractHistogramDiscretizedKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index_ref = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index_ref >= 0) {
    if (PARENT_USE_16BIT_HIST) {
      const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int32_t* larger_leaf_hist = reinterpret_cast<int32_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    } else if (LARGER_USE_16BIT_HIST) {
      int32_t* buffer = reinterpret_cast<int32_t*>(num_bit_change_buffer);
      const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        const int64_t parent_hist_item = larger_leaf_hist[global_thread_index];
        const int32_t smaller_hist_item = smaller_leaf_hist[global_thread_index];
        const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
          static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
        const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
        buffer[global_thread_index] = static_cast<int32_t>(static_cast<int16_t>(larger_hist_item >> 32) << 16) |
          static_cast<int32_t>(larger_hist_item & 0x000000000000ffff);
      }
    } else if (SMALLER_USE_16BIT_HIST) {
        const int32_t* smaller_leaf_hist = reinterpret_cast<const int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
        int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
        if (global_thread_index < num_total_bin) {
          const int64_t parent_hist_item = larger_leaf_hist[global_thread_index];
          const int32_t smaller_hist_item = smaller_leaf_hist[global_thread_index];
          const int64_t smaller_hist_item_int64 = (static_cast<int64_t>(static_cast<int16_t>(smaller_hist_item >> 16)) << 32) |
            static_cast<int64_t>(smaller_hist_item & 0x0000ffff);
          const int64_t larger_hist_item = parent_hist_item - smaller_hist_item_int64;
          larger_leaf_hist[global_thread_index] = larger_hist_item;
        }
    } else {
      const int64_t* smaller_leaf_hist = reinterpret_cast<const int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf);
      int64_t* larger_leaf_hist = reinterpret_cast<int64_t*>(cuda_larger_leaf_splits->hist_in_leaf);
      if (global_thread_index < num_total_bin) {
        larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
      }
    }
  }
}

__global__ void CopyChangedNumBitHistogram(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  hist_t* num_bit_change_buffer) {
  int32_t* hist_dst = reinterpret_cast<int32_t*>(cuda_larger_leaf_splits->hist_in_leaf);
  const int32_t* hist_src = reinterpret_cast<const int32_t*>(num_bit_change_buffer);
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_thread_index < static_cast<unsigned int>(num_total_bin)) {
    hist_dst[global_thread_index] = hist_src[global_thread_index];
  }
}

template <bool USE_16BIT_HIST>
__global__ void FixHistogramDiscretizedKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ int64_t shared_mem_buffer[WARPSIZE];
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  if (USE_16BIT_HIST) {
    const int64_t leaf_sum_gradients_hessians_int64 = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    const int32_t leaf_sum_gradients_hessians =
      (static_cast<int32_t>(leaf_sum_gradients_hessians_int64 >> 32) << 16) | static_cast<int32_t>(leaf_sum_gradients_hessians_int64 & 0x000000000000ffff);
    int32_t* feature_hist = reinterpret_cast<int32_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    const int32_t bin_gradient_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[threadIdx_x] : 0;
    const int32_t sum_gradient_hessian = ShuffleReduceSum<int32_t>(
      bin_gradient_hessian,
      reinterpret_cast<int32_t*>(shared_mem_buffer),
      num_bin_aligned);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  } else {
    const int64_t leaf_sum_gradients_hessians = cuda_smaller_leaf_splits->sum_of_gradients_hessians;
    int64_t* feature_hist = reinterpret_cast<int64_t*>(cuda_smaller_leaf_splits->hist_in_leaf) + feature_hist_offset;
    const unsigned int threadIdx_x = threadIdx.x;
    const uint32_t num_bin = cuda_feature_num_bins[feature_index];
    const int64_t bin_gradient_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[threadIdx_x] : 0;
    const int64_t sum_gradient_hessian = ShuffleReduceSum<int64_t>(bin_gradient_hessian, shared_mem_buffer, num_bin_aligned);
    if (threadIdx_x == 0) {
      feature_hist[most_freq_bin] = leaf_sum_gradients_hessians - sum_gradient_hessian;
    }
  }
}

void CUDAHistogramConstructor::LaunchSubtractHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool use_discretized_grad,
  const uint8_t parent_num_bits_in_histogram_bins,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
    if (!use_discretized_grad) {
      const int num_subtract_threads = 2 * num_total_bin_;
      const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
      global_timer.Start("CUDAHistogramConstructor::FixHistogramKernel");
      if (need_fix_histogram_features_.size() > 0) {
        FixHistogramKernel<<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
          cuda_feature_num_bins_.RawData(),
          cuda_feature_hist_offsets_.RawData(),
          cuda_feature_most_freq_bins_.RawData(),
          cuda_need_fix_histogram_features_.RawData(),
          cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
          cuda_smaller_leaf_splits);
      }
      global_timer.Stop("CUDAHistogramConstructor::FixHistogramKernel");
      global_timer.Start("CUDAHistogramConstructor::SubtractHistogramKernel");
      SubtractHistogramKernel<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
        num_total_bin_,
        cuda_smaller_leaf_splits,
        cuda_larger_leaf_splits);
      global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramKernel");
    } else {
      const int num_subtract_threads = num_total_bin_;
      const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
      global_timer.Start("CUDAHistogramConstructor::FixHistogramDiscretizedKernel");
      if (need_fix_histogram_features_.size() > 0) {
        if (smaller_num_bits_in_histogram_bins <= 16) {
          FixHistogramDiscretizedKernel<true><<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
            cuda_feature_num_bins_.RawData(),
            cuda_feature_hist_offsets_.RawData(),
            cuda_feature_most_freq_bins_.RawData(),
            cuda_need_fix_histogram_features_.RawData(),
            cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
            cuda_smaller_leaf_splits);
        } else {
          FixHistogramDiscretizedKernel<false><<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
            cuda_feature_num_bins_.RawData(),
            cuda_feature_hist_offsets_.RawData(),
            cuda_feature_most_freq_bins_.RawData(),
            cuda_need_fix_histogram_features_.RawData(),
            cuda_need_fix_histogram_features_num_bin_aligned_.RawData(),
            cuda_smaller_leaf_splits);
        }
      }
      global_timer.Stop("CUDAHistogramConstructor::FixHistogramDiscretizedKernel");
      global_timer.Start("CUDAHistogramConstructor::SubtractHistogramDiscretizedKernel");
      if (parent_num_bits_in_histogram_bins <= 16) {
        CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
        CHECK_LE(larger_num_bits_in_histogram_bins, 16);
        SubtractHistogramDiscretizedKernel<true, true, true><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          hist_buffer_for_num_bit_change_.RawData());
      } else if (larger_num_bits_in_histogram_bins <= 16) {
        CHECK_LE(smaller_num_bits_in_histogram_bins, 16);
        SubtractHistogramDiscretizedKernel<true, true, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          hist_buffer_for_num_bit_change_.RawData());
        CopyChangedNumBitHistogram<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          cuda_larger_leaf_splits,
          hist_buffer_for_num_bit_change_.RawData());
      } else if (smaller_num_bits_in_histogram_bins <= 16) {
        SubtractHistogramDiscretizedKernel<true, false, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          hist_buffer_for_num_bit_change_.RawData());
      } else {
        SubtractHistogramDiscretizedKernel<false, false, false><<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
          num_total_bin_,
          cuda_smaller_leaf_splits,
          cuda_larger_leaf_splits,
          hist_buffer_for_num_bit_change_.RawData());
      }
      global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramDiscretizedKernel");
    }
}

}  // namespace LightGBM

#endif  // USE_CUDA
