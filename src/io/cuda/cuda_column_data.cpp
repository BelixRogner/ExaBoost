/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_column_data.hpp>

#include <cstdint>
#include <vector>

namespace LightGBM {

CUDAColumnData::CUDAColumnData(const data_size_t num_data, const int gpu_device_id) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = num_data;
  gpu_device_id_ = gpu_device_id >= 0 ? gpu_device_id : 0;
  SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  data_by_column_.clear();
}

CUDAColumnData::~CUDAColumnData() {}

template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
void CUDAColumnData::InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, CUDAVector<uint8_t>* out_column_data_pointer) {
  CUDAVector<BIN_TYPE> cuda_column_data;
  if (!IS_SPARSE) {
    if (IS_4BIT) {
      std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
      const BIN_TYPE* in_column_data_reintrepreted = reinterpret_cast<const BIN_TYPE*>(in_column_data);
      for (data_size_t i = 0; i < num_data_; ++i) {
        expanded_column_data[i] = static_cast<BIN_TYPE>((in_column_data_reintrepreted[i >> 1] >> ((i & 1) << 2)) & 0xf);
      }
      cuda_column_data.InitFromHostVector(expanded_column_data);
    } else {
      cuda_column_data.InitFromHostMemory(reinterpret_cast<const BIN_TYPE*>(in_column_data), static_cast<size_t>(num_data_));
    }
  } else {
    // need to iterate bin iterator
    std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
    for (data_size_t i = 0; i < num_data_; ++i) {
      expanded_column_data[i] = static_cast<BIN_TYPE>(bin_iterator->RawGet(i));
    }
    cuda_column_data.InitFromHostVector(expanded_column_data);
  }
  out_column_data_pointer->MoveFrom(cuda_column_data, sizeof(BIN_TYPE) * cuda_column_data.Size());
}

void CUDAColumnData::Init(const int num_columns,
                          const std::vector<const void*>& column_data,
                          const std::vector<BinIterator*>& column_bin_iterator,
                          const std::vector<uint8_t>& column_bit_type,
                          const std::vector<uint32_t>& feature_max_bin,
                          const std::vector<uint32_t>& feature_min_bin,
                          const std::vector<uint32_t>& feature_offset,
                          const std::vector<uint32_t>& feature_most_freq_bin,
                          const std::vector<uint32_t>& feature_default_bin,
                          const std::vector<uint8_t>& feature_missing_is_zero,
                          const std::vector<uint8_t>& feature_missing_is_na,
                          const std::vector<uint8_t>& feature_mfb_is_zero,
                          const std::vector<uint8_t>& feature_mfb_is_na,
                          const std::vector<int>& feature_to_column) {
  num_columns_ = num_columns;
  column_bit_type_ = column_bit_type;
  feature_max_bin_ = feature_max_bin;
  feature_min_bin_ = feature_min_bin;
  feature_offset_ = feature_offset;
  feature_most_freq_bin_ = feature_most_freq_bin;
  feature_default_bin_ = feature_default_bin;
  feature_missing_is_zero_ = feature_missing_is_zero;
  feature_missing_is_na_ = feature_missing_is_na;
  feature_mfb_is_zero_ = feature_mfb_is_zero;
  feature_mfb_is_na_ = feature_mfb_is_na;
  // Decide whether to skip the per-column GPU allocation. With a 32 GB GPU and
  // a 17 GB row matrix on top, we can't afford another 17 GB of per-column data.
  // Skip when total size would exceed 8 GB; the caller (tree learner) will provide
  // a compact column view per tree via SetCompactColumnView.
  size_t expected_total_bytes = 0;
  for (int c = 0; c < num_columns_; ++c) {
    const int8_t bt = column_bit_type[c];
    int bytes_per = (bt == 4 || bt == 8) ? 1 : (bt == 16 ? 2 : 4);
    expected_total_bytes += static_cast<size_t>(num_data_) * bytes_per;
  }
  init_skipped_per_column_alloc_ = (expected_total_bytes > static_cast<size_t>(8) * 1024 * 1024 * 1024);
  if (init_skipped_per_column_alloc_) {
    Log::Warning("CUDAColumnData: skipping per-column allocation (would be %.2f GB). "
                 "Caller must invoke SetCompactColumnView per tree.", expected_total_bytes / 1e9);
    // Safety: ensure GetColumnData() has a valid (null-filled) host view until
    // SetCompactColumnView populates it for the first tree.
    compact_column_host_view_.assign(num_columns_, nullptr);
  }
  for (int column_index = 0; column_index < num_columns_; ++column_index) {
    data_by_column_.emplace_back(new CUDAVector<uint8_t>());
  }
  OMP_INIT_EX();
  #pragma omp parallel num_threads(num_threads_)
  {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
    #pragma omp for schedule(static)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const int8_t bit_type = column_bit_type[column_index];
      if (column_data[column_index] != nullptr) {
        // is dense column
        if (init_skipped_per_column_alloc_) {
          // Adjust column_bit_type_ for 4-bit case (which expanded to 8) and skip GPU alloc.
          if (bit_type == 4) {
            column_bit_type_[column_index] = 8;
          }
          data_by_column_[column_index] = nullptr;
        } else if (bit_type == 4) {
          column_bit_type_[column_index] = 8;
          InitOneColumnData<false, true, uint8_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else if (bit_type == 8) {
          InitOneColumnData<false, false, uint8_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else if (bit_type == 16) {
          InitOneColumnData<false, false, uint16_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else if (bit_type == 32) {
          InitOneColumnData<false, false, uint32_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else {
          Log::Fatal("Unknown column bit type %d", bit_type);
        }
      } else {
        // is sparse column
        if (bit_type == 8) {
          InitOneColumnData<true, false, uint8_t>(nullptr, column_bin_iterator[column_index], data_by_column_[column_index].get());
        } else if (bit_type == 16) {
          InitOneColumnData<true, false, uint16_t>(nullptr, column_bin_iterator[column_index], data_by_column_[column_index].get());
        } else if (bit_type == 32) {
          InitOneColumnData<true, false, uint32_t>(nullptr, column_bin_iterator[column_index], data_by_column_[column_index].get());
        } else {
          Log::Fatal("Unknown column bit type %d", bit_type);
        }
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  feature_to_column_ = feature_to_column;
  cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
  InitColumnMetaInfo();
}

void CUDAColumnData::CopySubrow(
  const CUDAColumnData* full_set,
  const data_size_t* used_indices,
  const data_size_t num_used_indices) {
  num_threads_ = full_set->num_threads_;
  num_columns_ = full_set->num_columns_;
  column_bit_type_ = full_set->column_bit_type_;
  feature_min_bin_ = full_set->feature_min_bin_;
  feature_max_bin_ = full_set->feature_max_bin_;
  feature_offset_ = full_set->feature_offset_;
  feature_most_freq_bin_ = full_set->feature_most_freq_bin_;
  feature_default_bin_ = full_set->feature_default_bin_;
  feature_missing_is_zero_ = full_set->feature_missing_is_zero_;
  feature_missing_is_na_ = full_set->feature_missing_is_na_;
  feature_mfb_is_zero_ = full_set->feature_mfb_is_zero_;
  feature_mfb_is_na_ = full_set->feature_mfb_is_na_;
  feature_to_column_ = full_set->feature_to_column_;
  // Propagate the Tier-2 skip flag from the full set. When the full set's
  // per-column GPU buffers were skipped (e.g. >8GB total — Init() decides),
  // its data_by_column_ entries are empty. The CopySubrow kernel cannot read
  // from them, so the subset must also skip per-column allocation and rely on
  // the compact-view system (BuildCompactColumnView / SetCompactColumnView),
  // which reads from the subset's CUDARowData and therefore naturally produces
  // subset-sized buffers per tree.
  init_skipped_per_column_alloc_ = full_set->init_skipped_per_column_alloc_;
  if (cuda_used_indices_.Size() == 0) {
    // initialize the subset cuda column data
    const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
    cuda_used_indices_.Resize(num_used_indices_size);
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      data_by_column_.emplace_back(new CUDAVector<uint8_t>());
    }
    if (!init_skipped_per_column_alloc_) {
      OMP_INIT_EX();
      #pragma omp parallel num_threads(num_threads_)
      {
        SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
        #pragma omp for schedule(static)
        for (int column_index = 0; column_index < num_columns_; ++column_index) {
          OMP_LOOP_EX_BEGIN();
          const uint8_t bit_type = column_bit_type_[column_index];
          if (bit_type == 8) {
            CUDAVector<uint8_t> column_data;
            column_data.Resize(num_used_indices_size);
            data_by_column_[column_index]->MoveFrom(column_data, sizeof(uint8_t) * column_data.Size());
          } else if (bit_type == 16) {
            CUDAVector<uint16_t> column_data;
            column_data.Resize(num_used_indices_size);
            data_by_column_[column_index]->MoveFrom(column_data, sizeof(uint16_t) * column_data.Size());
          } else if (bit_type == 32) {
            CUDAVector<uint32_t> column_data;
            column_data.Resize(num_used_indices_size);
            data_by_column_[column_index]->MoveFrom(column_data, sizeof(uint32_t) * column_data.Size());
          }
          OMP_LOOP_EX_END();
        }
      }
      OMP_THROW_EX();
    }
    cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
    InitColumnMetaInfo();
    cur_subset_buffer_size_ = num_used_indices;
  } else {
    if (num_used_indices > cur_subset_buffer_size_) {
      ResizeWhenCopySubrow(num_used_indices);
      cur_subset_buffer_size_ = num_used_indices;
    }
  }
  cuda_used_indices_.InitFromHostMemory(used_indices, static_cast<size_t>(num_used_indices));
  num_used_indices_ = num_used_indices;
  // In the skipped path full_set has no per-column buffers to copy from; the
  // subset's per-tree buffers come from the compact-view system instead.
  if (!init_skipped_per_column_alloc_) {
    LaunchCopySubrowKernel(full_set->cuda_data_by_column());
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAColumnData::SetCompactColumnView(const std::vector<int>& column_to_compact_slot,
                                          void* compact_buf,
                                          size_t bytes_per_col) {
  // Repoint cuda_data_by_column_ at the per-tree compact buffer. data_by_column_
  // (the owning unique_ptr<CUDAVector<uint8_t>>s) is left untouched; we only
  // override the GPU-side pointer table that kernels read.
  std::vector<uint8_t*> view(num_columns_, nullptr);
  uint8_t* base = reinterpret_cast<uint8_t*>(compact_buf);
  for (int c = 0; c < num_columns_; ++c) {
    if (c < static_cast<int>(column_to_compact_slot.size()) && column_to_compact_slot[c] >= 0) {
      const size_t off = static_cast<size_t>(column_to_compact_slot[c]) * bytes_per_col;
      view[c] = base + off;
    }
  }
  cuda_data_by_column_.InitFromHostVector(view);
  // Keep a host-readable mirror so GetColumnData() can return the split column's
  // device pointer in the skip-allocation path (data_by_column_ stays null there).
  compact_column_host_view_ = view;
}

void CUDAColumnData::RestoreOriginalColumnView() {
  // Restore cuda_data_by_column_ to point at the original per-column
  // allocations owned by data_by_column_. No-op when init_skipped_per_column_alloc_
  // was set (the original "view" is just nullptrs, which is fine as long as
  // no kernel reads it before another SetCompactColumnView).
  cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
}

void CUDAColumnData::ResizeWhenCopySubrow(const data_size_t num_used_indices) {
  const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
  cuda_used_indices_.Resize(num_used_indices_size);
  OMP_INIT_EX();
  #pragma omp parallel num_threads(num_threads_)
  {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
    #pragma omp for schedule(static)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const uint8_t bit_type = column_bit_type_[column_index];
      if (bit_type == 8) {
        data_by_column_[column_index]->Resize(sizeof(uint8_t) * num_used_indices_size);
      } else if (bit_type == 16) {
        data_by_column_[column_index]->Resize(sizeof(uint16_t) * num_used_indices_size);
      } else if (bit_type == 32) {
        data_by_column_[column_index]->Resize(sizeof(uint32_t) * num_used_indices_size);
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
}

void CUDAColumnData::InitColumnMetaInfo() {
  cuda_column_bit_type_.InitFromHostVector(column_bit_type_);
  cuda_feature_max_bin_.InitFromHostVector(feature_max_bin_);
  cuda_feature_min_bin_.InitFromHostVector(feature_min_bin_);
  cuda_feature_offset_.InitFromHostVector(feature_offset_);
  cuda_feature_most_freq_bin_.InitFromHostVector(feature_most_freq_bin_);
  cuda_feature_default_bin_.InitFromHostVector(feature_default_bin_);
  cuda_feature_missing_is_zero_.InitFromHostVector(feature_missing_is_zero_);
  cuda_feature_missing_is_na_.InitFromHostVector(feature_missing_is_na_);
  cuda_feature_mfb_is_zero_.InitFromHostVector(feature_mfb_is_zero_);
  cuda_feature_mfb_is_na_.InitFromHostVector(feature_mfb_is_na_);
  cuda_feature_to_column_.InitFromHostVector(feature_to_column_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
