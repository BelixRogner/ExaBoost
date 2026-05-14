/*!
 * Copyright (c) 2026 ExaBoost authors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_METAL_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_METAL_TREE_LEARNER_H_

#include "serial_tree_learner.h"

#ifdef USE_METAL

#include <memory>
#include <vector>

namespace MTL {
class Device;
class CommandQueue;
class Library;
class ComputePipelineState;
class Buffer;
}  // namespace MTL

namespace LightGBM {

/*!
 * \brief Metal-accelerated tree learner for Apple silicon GPUs.
 *
 * Subclasses SerialTreeLearner. Overrides ConstructHistograms to run the
 * gradient/hessian histogram construction on the Apple GPU via the kernels
 * in src/treelearner/metal/. Non-eligible feature groups (sparse, non-256-bin)
 * fall back to the CPU path.
 */
class MetalTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalTreeLearner(const Config* tree_config);
  ~MetalTreeLearner() override;

  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian,
                              bool reset_multi_val_bin) override;

 protected:
  void BeforeTrain() override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                           bool use_subtract) override;

 private:
  void InitMetal();
  void TeardownMetal();
  bool BuildDenseFeatureBuffer();  // copies eligible dense features into a uchar buffer
  void RunMetalHistogram(const score_t* gradients, const score_t* hessians,
                         data_size_t num_data);
  void RunMetalHistogramIndexed(const data_size_t* data_indices, data_size_t num_idx);
  // Sibling-batching variant: dispatch BOTH smaller-leaf and larger-leaf
  // histograms in one command buffer + one waitUntilCompleted. Used for the
  // use_subtract=false path where we'd otherwise pay sync overhead twice.
  void RunMetalHistogramSibling(
      const data_size_t* smaller_indices, data_size_t smaller_n,
      const data_size_t* larger_indices,  data_size_t larger_n);
  // Quantized variants: read int8 packed grad+hess, produce int32 histograms.
  // Caller is responsible for staging gh_packed_buf before calling.
  void RunMetalHistogramQ32(data_size_t num_data);
  void RunMetalHistogramQ32Indexed(const data_size_t* data_indices, data_size_t num_idx);

  struct MetalState;
  std::unique_ptr<MetalState> state_;

  /*! \brief True after BuildDenseFeatureBuffer has materialized this dataset's
   * feature buffer; gates the Metal path in ConstructHistograms. */
  bool metal_buffer_ready_ = false;
  /*! \brief Cached per-feature bin counts (avoids virtual calls in hot write-back loop). */
  std::vector<int> per_feature_num_bin_;
  /*! \brief Cached per-feature offset (0 or 1 depending on most-frequent bin). */
  std::vector<int> per_feature_offset_;
  /*! \brief True when the Metal device + kernels + queue are ready. */
  bool metal_ready_ = false;
};

}  // namespace LightGBM

#else  // !USE_METAL

namespace LightGBM {

class MetalTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalTreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
    Log::Fatal("Metal Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DUSE_METAL=ON.");
  }
};

}  // namespace LightGBM

#endif  // USE_METAL

#endif  // LIGHTGBM_TREELEARNER_METAL_TREE_LEARNER_H_
