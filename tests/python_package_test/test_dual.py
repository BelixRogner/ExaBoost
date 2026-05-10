# coding: utf-8
"""Tests for dual GPU+CPU support."""

import contextlib
import io
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


def _get_init_score(device_type, objective, alpha, X, y):
    """Train a 1-tree model and read 'Start training from score' from the log."""
    params = {
        "objective": objective,
        "alpha": alpha,
        "verbose": 1,
        "num_leaves": 2,
        "min_data_in_leaf": 1,
        "learning_rate": 0.1,
        "deterministic": True,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "seed": 0,
        "device_type": device_type,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        lgb.train(params, ds, num_boost_round=1)
    for line in buf.getvalue().splitlines():
        if "Start training from score" in line:
            return float(line.split("score")[-1].strip())
    raise AssertionError(f"no init score logged for {device_type} {objective} alpha={alpha}")


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "alpha"), [("regression_l1", 0.5), ("quantile", 0.5), ("quantile", 0.3), ("quantile", 0.7)]
)
@pytest.mark.parametrize("n", [5, 7, 10, 11, 100, 500])
def test_cuda_init_score_matches_cpu(objective, alpha, n):
    """CUDA percentile-based init scores must match CPU at FP epsilon.

    Regression test for the bug in PercentileGlobalKernel that used
    `(1 - alpha) * len` instead of `(1 - alpha) * (len - 1)`. For
    objective=regression_l1 with y=[1..5], CUDA returned 3.5 instead of
    the correct 3.0.
    """
    X = np.zeros((n, 1))
    y = np.arange(1, n + 1, dtype=np.float64)
    cpu = _get_init_score("cpu", objective, alpha, X, y)
    cuda = _get_init_score("cuda", objective, alpha, X, y)
    assert cuda == pytest.approx(cpu, abs=1e-6), f"{objective} alpha={alpha} n={n}: cpu={cpu} cuda={cuda}"


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
