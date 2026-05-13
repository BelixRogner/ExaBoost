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

  struct MetalState;
  std::unique_ptr<MetalState> state_;

  /*! \brief Feature groups eligible for Metal acceleration (dense, 256-bin). */
  std::vector<int> metal_feature_groups_;
  /*! \brief True when a Metal pipeline is fully ready for histogram construction. */
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
