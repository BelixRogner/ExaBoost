/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/tree_learner.h>

#include <functional>
#include <string>
#include <unordered_map>

#include "gpu_tree_learner.h"
#include "linear_tree_learner.h"
#include "parallel_tree_learner.h"
#include "serial_tree_learner.h"
#include "cuda/cuda_single_gpu_tree_learner.hpp"

namespace LightGBM {

namespace {

// Build the standard serial / parallel / linear family on top of a given base
// learner. CPU and OpenCL share this exact structure -- only the base learner
// differs -- so it lives here once instead of being copy-pasted per backend.
template <typename Base>
TreeLearner* MakeSerialFamily(const std::string& learner_type, const Config* config) {
  if (learner_type == std::string("serial")) {
    if (config->linear_tree) {
      return new LinearTreeLearner<Base>(config);
    }
    return new Base(config);
  } else if (learner_type == std::string("feature")) {
    return new FeatureParallelTreeLearner<Base>(config);
  } else if (learner_type == std::string("data")) {
    return new DataParallelTreeLearner<Base>(config);
  } else if (learner_type == std::string("voting")) {
    return new VotingParallelTreeLearner<Base>(config);
  }
  return nullptr;
}

}  // namespace

TreeLearner* TreeLearner::CreateTreeLearner(const std::string& learner_type, const std::string& device_type,
                                            const Config* config, const bool boosting_on_cuda) {
  // Backend SPI. Each entry maps a device_type to a factory that produces a
  // TreeLearner (the backend interface). Adding a new backend is a one-line
  // registration here plus an implementation of the TreeLearner interface --
  // no other call site needs to change.
  using Factory = std::function<TreeLearner*(const std::string& /*learner_type*/,
                                             const Config* /*config*/, bool /*boosting_on_cuda*/)>;
  static const std::unordered_map<std::string, Factory> registry = {
    {"cpu", [](const std::string& lt, const Config* c, bool) {
      return MakeSerialFamily<SerialTreeLearner>(lt, c);
    }},
    {"gpu", [](const std::string& lt, const Config* c, bool) {
      return MakeSerialFamily<GPUTreeLearner>(lt, c);
    }},
    {"cuda", [](const std::string& lt, const Config* c, bool on_cuda) -> TreeLearner* {
      if (lt == std::string("serial")) {
        return new CUDASingleGPUTreeLearner(c, on_cuda);
      }
      Log::Fatal("Currently cuda version only supports training on a single machine.");
      return nullptr;
    }},
  };
  const auto it = registry.find(device_type);
  if (it == registry.end()) {
    return nullptr;
  }
  return it->second(learner_type, config, boosting_on_cuda);
}

}  // namespace LightGBM
