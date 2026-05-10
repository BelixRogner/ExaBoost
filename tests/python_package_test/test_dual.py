# coding: utf-8
"""Tests for dual GPU+CPU support."""

import os
import platform

import numpy as np
import pytest
from sklearn.metrics import log_loss

import lightgbm as lgb

from .utils import load_breast_cancer


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression_l1", "quantile"])
@pytest.mark.parametrize("n", [100, 200, 500, 1000])
def test_cuda_weighted_percentile_renewal_does_not_crash(objective, n):
    """Regression test for the OOB shared-memory access in
    ShuffleSortedPrefixSumDevice that crashed weighted L1 / weighted
    quantile training with "illegal memory access" for n >= ~100.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3)).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)
    w = rng.random(n)
    ds = lgb.Dataset(X, label=y, weight=w, params={"verbose": -1, "feature_pre_filter": False})
    params = {
        "objective": objective,
        "alpha": 0.5,
        "device_type": "cuda",
        "verbose": -1,
        "num_leaves": 4,
        "min_data_in_leaf": 1,
        "deterministic": True,
        "gpu_use_dp": True,
    }
    # If the OOB access regresses, this raises a CUDA "illegal memory access" error.
    bst = lgb.train(params, ds, num_boost_round=2)
    preds = bst.predict(X, raw_score=True)
    assert np.all(np.isfinite(preds)), "weighted percentile renewal produced non-finite predictions"


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", "0") != "1",
    reason="Set LIGHTGBM_TEST_DUAL_CPU_GPU=1 to test using CPU and GPU training from the same package.",
)
def test_cpu_and_gpu_work():
    # If compiled appropriately, the same installation will support both GPU and CPU.
    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, y)

    params_cpu = {"verbosity": -1, "num_leaves": 31, "objective": "binary", "device": "cpu"}
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
