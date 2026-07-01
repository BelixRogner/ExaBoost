/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Device-agnostic core of the linear-tree leaf fit. The per-leaf normal
 * equations (XtHX, Xtg) are accumulated per backend -- an OMP loop on CPU, a
 * CUDA kernel on GPU -- but the ridge-regularized solve, coefficient pruning,
 * and storage onto the Tree are identical, so they live here once and are
 * shared by LinearTreeLearner (CPU/OpenCL) and the CUDA tree learner.
 */
#ifndef LIGHTGBM_TREELEARNER_LINEAR_LEAF_SOLVER_H_
#define LIGHTGBM_TREELEARNER_LINEAR_LEAF_SOLVER_H_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>

#include <Eigen/Dense>

#include <vector>

namespace LightGBM {

namespace LinearLeafSolver {

// Solve coeffs = -(XtHX + lambda*I_features)^-1 * Xtg for one leaf, prune
// near-zero coefficients (non-refit) or blend with the old model (refit), and
// store the resulting linear model onto the tree.
//   xthx: upper-triangle (row-major) of the (num_feat+1)x(num_feat+1) gram,
//         length (num_feat+1)*(num_feat+2)/2
//   xtg:  length num_feat+1 (last entry is the constant term)
//   features_inner: num_feat inner feature indices, paired with the first
//                   num_feat coefficients
inline void SolveAndStore(Tree* tree, int leaf, int num_feat,
                          const double* xthx, const double* xtg,
                          const std::vector<int>& features_inner,
                          const Dataset* train_data, double lambda,
                          bool is_refit, double decay_rate, double shrinkage,
                          const std::vector<int>& old_coeff_idx) {
  Eigen::MatrixXd XTHX_mat(num_feat + 1, num_feat + 1);
  Eigen::MatrixXd XTg_mat(num_feat + 1, 1);
  int j = 0;
  for (int feat1 = 0; feat1 < num_feat + 1; ++feat1) {
    for (int feat2 = feat1; feat2 < num_feat + 1; ++feat2) {
      XTHX_mat(feat1, feat2) = xthx[j];
      XTHX_mat(feat2, feat1) = XTHX_mat(feat1, feat2);
      if ((feat1 == feat2) && (feat1 < num_feat)) {
        XTHX_mat(feat1, feat2) += lambda;
      }
      ++j;
    }
    XTg_mat(feat1) = xtg[feat1];
  }
  Eigen::MatrixXd coeffs = -XTHX_mat.fullPivLu().inverse() * XTg_mat;
  std::vector<double> coeffs_vec;
  std::vector<int> features_new;
  const std::vector<double> old_coeffs = tree->LeafCoeffs(leaf);
  for (int i = 0; i < num_feat; ++i) {
    if (is_refit) {
      features_new.push_back(features_inner[i]);
      // old_coeff_idx maps position i back to the original coefficient, since
      // some features may have been skipped (absent from the refit dataset).
      const int old_i = old_coeff_idx[i];
      coeffs_vec.push_back(decay_rate * old_coeffs[old_i] + (1.0 - decay_rate) * coeffs(i) * shrinkage);
    } else {
      if (coeffs(i) < -kZeroThreshold || coeffs(i) > kZeroThreshold) {
        coeffs_vec.push_back(coeffs(i));
        features_new.push_back(features_inner[i]);
      }
    }
  }
  tree->SetLeafFeaturesInner(leaf, features_new);
  std::vector<int> features_raw(features_new.size());
  for (size_t i = 0; i < features_new.size(); ++i) {
    features_raw[i] = train_data->RealFeatureIndex(features_new[i]);
  }
  tree->SetLeafFeatures(leaf, features_raw);
  tree->SetLeafCoeffs(leaf, coeffs_vec);
  if (is_refit) {
    const double old_const = tree->LeafConst(leaf);
    tree->SetLeafConst(leaf, decay_rate * old_const + (1.0 - decay_rate) * coeffs(num_feat) * shrinkage);
  } else {
    tree->SetLeafConst(leaf, coeffs(num_feat));
  }
}

}  // namespace LinearLeafSolver

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_LINEAR_LEAF_SOLVER_H_
