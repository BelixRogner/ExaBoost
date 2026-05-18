# coding: utf-8
"""Tests for dual GPU+CPU support."""

import os
import platform

import numpy as np
import pytest
from sklearn.metrics import log_loss

import lightgbm as lgb

from .utils import load_breast_cancer


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", "0") != "1",
    reason="Set LIGHTGBM_TEST_DUAL_CPU_GPU=1 to test using CPU and GPU training from the same package.",
)
def test_cpu_and_gpu_work():
    # If compiled appropriately, the same installation will support both GPU and CPU.
    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, y)

    params_cpu = {
        "verbosity": -1,
        "num_leaves": 31,
        "objective": "binary",
        "device": "cpu",
    }
    cpu_bst = lgb.train(params_cpu, data, num_boost_round=10)
    cpu_score = log_loss(y, cpu_bst.predict(X))

    params_gpu = params_cpu.copy()
    params_gpu["device"] = "gpu"
    # Double-precision floats are only supported on x86_64 with PoCL
    params_gpu["gpu_use_dp"] = platform.machine() == "x86_64"
    gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
    gpu_score = log_loss(y, gpu_bst.predict(X))

    rel = 1e-6 if params_gpu["gpu_use_dp"] else 1e-4
    assert cpu_score == pytest.approx(gpu_score, rel=rel)
    assert gpu_score < 0.242


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


def _make_regression_for_parity(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    coef = rng.standard_normal(d)
    y = (X @ coef + 0.1 * rng.standard_normal(n)).astype(np.float64)
    return X, y


def _train_cpu_and_cuda(params_overrides, X, y, num_round):
    out = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 42,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "min_data_in_leaf": 5,
            "min_sum_hessian_in_leaf": 1e-3,
            **params_overrides,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        out[device_type] = lgb.train(params, ds, num_boost_round=num_round)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("name", "params_overrides", "seed", "num_round"),
    [
        # Regression test for the gain-plateau argmax bug. Bagging configuration
        # was the original failure: max|Δ|=0.39 at round 3, structurally
        # divergent trees from round 3 onward. With the fix, all 5 rounds
        # match at fp64 epsilon and trees are bit-identical.
        (
            "bagging",
            {"bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 1},
            11,
            5,
        ),
        # Plain dense regression: predictions matched at fp64 epsilon prior
        # to the fix but the encoded tree thresholds differed cosmetically
        # (CPU and CUDA picked different bins from a true gain plateau).
        # After the fix, trees are bit-identical.
        ("dense", {}, 1, 5),
        # max_depth regression: another configuration where round-1 trees
        # had cosmetic threshold-encoding differences before the fix.
        ("max_depth", {"max_depth": 3}, 12, 5),
        # L2 regularisation: same family.
        ("l2", {"lambda_l2": 1.0}, 7, 5),
    ],
)
def test_cuda_split_gain_tie_break_matches_cpu(name, params_overrides, seed, num_round):
    """CUDA must match CPU at fp64 epsilon when the best-split argmax has a
    gain plateau (multiple bins with truly equal gain).

    Prior to the tolerance-based tie-break in cuda_best_split_finder.cu's
    ReduceBestGain* helpers, ULP-level FP noise in the parallel histogram
    flipped which bin from the plateau had the slightly-higher numerical
    gain on CUDA. CPU's exact computation picked a different bin (its
    sequential scan + strict ``>`` comparison resolves to the lowest-index
    bin on the plateau). The threshold-encoding mismatch was at round 1
    cosmetic for predictions (data routed identically), but compounded
    through score updates and surfaced as structural tree divergence by
    round 3 in cases like reg_bagging.
    """
    X, y = _make_regression_for_parity(seed=seed)
    pair = _train_cpu_and_cuda(params_overrides, X, y, num_round=num_round)
    pred_cpu = pair["cpu"].predict(X, raw_score=True)
    pred_cuda = pair["cuda"].predict(X, raw_score=True)
    # fp64 epsilon ≈ 2.2e-16; allow a generous 1e-10 to absorb any
    # remaining round-by-round drift from sources unrelated to this fix.
    np.testing.assert_allclose(pred_cuda, pred_cpu, atol=1e-10)
