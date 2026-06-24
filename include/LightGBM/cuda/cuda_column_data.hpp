/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_COLUMN_DATA_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_COLUMN_DATA_HPP_

#ifdef USE_CUDA

#include <LightGBM/config.h>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/bin.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <memory>
#include <cstdint>
#include <vector>

namespace LightGBM {

class CUDAColumnData {
 public:
  CUDAColumnData(const data_size_t num_data, const int gpu_device_id);

  ~CUDAColumnData();

  void Init(const int num_columns,
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
            const std::vector<int>& feature_to_column);

  const uint8_t* GetColumnData(const int column_index) const {
    // In the skip-allocation path, data_by_column_[c] is null; the per-column
    // device pointers live in the per-tree compact view (set by
    // SetCompactColumnView). Read those instead to avoid a host null-deref.
    if (init_skipped_per_column_alloc_) {
      return compact_column_host_view_[column_index];
    }
    return data_by_column_[column_index]->RawData();
  }

  void CopySubrow(const CUDAColumnData* full_set, const data_size_t* used_indices, const data_size_t num_used_indices);

  uint8_t* const* cuda_data_by_column() const { return cuda_data_by_column_.RawData(); }

  uint32_t feature_min_bin(const int feature_index) const { return feature_min_bin_[feature_index]; }

  uint32_t feature_max_bin(const int feature_index) const { return feature_max_bin_[feature_index]; }

  uint32_t feature_offset(const int feature_index) const { return feature_offset_[feature_index]; }

  uint32_t feature_most_freq_bin(const int feature_index) const { return feature_most_freq_bin_[feature_index]; }

  uint32_t feature_default_bin(const int feature_index) const { return feature_default_bin_[feature_index]; }

  uint8_t feature_missing_is_zero(const int feature_index) const { return feature_missing_is_zero_[feature_index]; }

  uint8_t feature_missing_is_na(const int feature_index) const { return feature_missing_is_na_[feature_index]; }

  uint8_t feature_mfb_is_zero(const int feature_index) const { return feature_mfb_is_zero_[feature_index]; }

  uint8_t feature_mfb_is_na(const int feature_index) const { return feature_mfb_is_na_[feature_index]; }

  const uint32_t* cuda_feature_min_bin() const { return cuda_feature_min_bin_.RawData(); }

  const uint32_t* cuda_feature_max_bin() const { return cuda_feature_max_bin_.RawData(); }

  const uint32_t* cuda_feature_offset() const { return cuda_feature_offset_.RawData(); }

  const uint32_t* cuda_feature_most_freq_bin() const { return cuda_feature_most_freq_bin_.RawData(); }

  const uint32_t* cuda_feature_default_bin() const { return cuda_feature_default_bin_.RawData(); }

  const uint8_t* cuda_feature_missing_is_zero() const { return cuda_feature_missing_is_zero_.RawData(); }

  const uint8_t* cuda_feature_missing_is_na() const { return cuda_feature_missing_is_na_.RawData(); }

  const uint8_t* cuda_feature_mfb_is_zero() const { return cuda_feature_mfb_is_zero_.RawData(); }

  const uint8_t* cuda_feature_mfb_is_na() const { return cuda_feature_mfb_is_na_.RawData(); }

  const int* cuda_feature_to_column() const { return cuda_feature_to_column_.RawData(); }

  const uint8_t* cuda_column_bit_type() const { return cuda_column_bit_type_.RawData(); }

  int feature_to_column(const int feature_index) const { return feature_to_column_[feature_index]; }

  uint8_t column_bit_type(const int column_index) const { return column_bit_type_[column_index]; }

  // ===== Per-tree compact column view =====
  // Set a compact column buffer for sampled features. Updates data_by_column_[c]
  // pointers for sampled features to point into compact_buf at strided offsets;
  // re-uploads cuda_data_by_column_. Non-sampled features get nullptr.
  // Compact layout: compact_buf[slot * num_data + r]; slot = compact index assigned
  // by the caller via column_to_compact_slot[c] (-1 if not in sample).
  void SetCompactColumnView(const std::vector<int>& column_to_compact_slot,
                            void* compact_buf,
                            size_t bytes_per_col);

  // Restore data_by_column_ to point to the original per-column allocations.
  // Needed when feature_fraction = 1.0 / no compaction (rare path).
  void RestoreOriginalColumnView();

  // Skip per-column allocation in Init? Used when caller will provide compact view.
  bool init_skipped_per_column_alloc_ = false;

 private:
  template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
  void InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, CUDAVector<uint8_t>* out_column_data_pointer);

  void LaunchCopySubrowKernel(uint8_t* const* in_cuda_data_by_column);

  void InitColumnMetaInfo();

  void ResizeWhenCopySubrow(const data_size_t num_used_indices);

  std::vector<uint8_t*> GetDataByColumnPointers(const std::vector<std::unique_ptr<CUDAVector<uint8_t>>>& data_by_column) const {
    std::vector<uint8_t*> data_by_column_pointers(data_by_column.size(), nullptr);
    for (size_t i = 0; i < data_by_column.size(); ++i) {
      // unique_ptr can be null when init_skipped_per_column_alloc_ is set —
      // those slots are filled later by SetCompactColumnView().
      if (data_by_column[i]) {
        data_by_column_pointers[i] = reinterpret_cast<uint8_t*>(data_by_column[i]->RawData());
      }
    }
    return data_by_column_pointers;
  }

  int gpu_device_id_;
  int num_threads_;
  data_size_t num_data_;
  int num_columns_;
  std::vector<uint8_t> column_bit_type_;
  std::vector<uint32_t> feature_min_bin_;
  std::vector<uint32_t> feature_max_bin_;
  std::vector<uint32_t> feature_offset_;
  std::vector<uint32_t> feature_most_freq_bin_;
  std::vector<uint32_t> feature_default_bin_;
  std::vector<uint8_t> feature_missing_is_zero_;
  std::vector<uint8_t> feature_missing_is_na_;
  std::vector<uint8_t> feature_mfb_is_zero_;
  std::vector<uint8_t> feature_mfb_is_na_;
  CUDAVector<uint8_t*> cuda_data_by_column_;
  std::vector<int> feature_to_column_;
  std::vector<std::unique_ptr<CUDAVector<uint8_t>>> data_by_column_;
  // Host mirror of the per-column device pointers when init_skipped_per_column_alloc_
  // is set: each entry points into the per-tree compact buffer (slot * num_data),
  // or nullptr for non-sampled columns. Populated by SetCompactColumnView so that
  // GetColumnData has a valid host-readable device pointer for the split column.
  std::vector<uint8_t*> compact_column_host_view_;

  CUDAVector<uint8_t> cuda_column_bit_type_;
  CUDAVector<uint32_t> cuda_feature_min_bin_;
  CUDAVector<uint32_t> cuda_feature_max_bin_;
  CUDAVector<uint32_t> cuda_feature_offset_;
  CUDAVector<uint32_t> cuda_feature_most_freq_bin_;
  CUDAVector<uint32_t> cuda_feature_default_bin_;
  CUDAVector<uint8_t> cuda_feature_missing_is_zero_;
  CUDAVector<uint8_t> cuda_feature_missing_is_na_;
  CUDAVector<uint8_t> cuda_feature_mfb_is_zero_;
  CUDAVector<uint8_t> cuda_feature_mfb_is_na_;
  CUDAVector<int> cuda_feature_to_column_;

  // used when bagging with subset
  CUDAVector<data_size_t> cuda_used_indices_;
  data_size_t num_used_indices_;
  data_size_t cur_subset_buffer_size_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_COLUMN_DATA_HPP_
