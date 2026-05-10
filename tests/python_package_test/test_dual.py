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
def test_cuda_lambdarank_round1_matches_cpu_within_fp_drift():
    """Regression test for non-stable BitonicArgSort with all-equal round-1 scores.

    Before the BitonicArgSort tie-stability fix, CUDA's bitonic sort swapped
    tied elements during descending sort because `(a > b) == false` was true
    for `a == b`. This shuffled the document order in each LambdaRank query
    on round 1 (when all scores are 0) and produced max|Δ|=0.29 on a small
    dataset. With stable ties, the same case drops to ~0.14 (residual is
    FP-precision in atomicAdd_block ordering).
    """
    rng = np.random.default_rng(42)
    n_queries = 10
    items_per_query = 20
    n = n_queries * items_per_query
    n_features = 8
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    w = rng.standard_normal(n_features)
    y = np.clip(np.round(X @ w + 0.5 * rng.standard_normal(n) + 2), 0, 4).astype(int)
    group = np.full(n_queries, items_per_query, dtype=np.int32)

    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 42,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
    }
    preds = {}
    for dev in ("cpu", "cuda"):
        ds = lgb.Dataset(X, label=y, group=group, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train({**base, "device_type": dev}, ds, num_boost_round=1)
        preds[dev] = bst.predict(X, raw_score=True)
    diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    # Tolerance is well below the 0.29 the pre-fix bug produced. After the fix
    # ~0.14 remains from FP-precision in pair-gradient atomic-add ordering
    # (documented expected behavior per upstream #6055), so we set the bar at
    # 0.2 — strict enough to catch the bitonic-sort regression, loose enough to
    # tolerate the FP-precision residual.
    assert diff < 0.2, f"LambdaRank round-1 max|Δ|={diff:.4e} (was ~0.29 before BitonicArgSort fix)"


@_REQUIRES_CUDA
def test_cuda_bitonic_argsort_1024_with_distinct_scores_matches_cpu():
    """Regression test for BitonicArgSort_1024's per-pass `ascending` direction.

    A first attempt at the tie-stability fix replaced the per-pass `ascending`
    local with the template parameter `ASCENDING`. That preserves correctness
    for all-tied inputs (LambdaRank round 1) because the strict comparator
    returns false either way, but breaks the bitonic merge for non-tied
    inputs because outer phases must alternate direction.

    BitonicArgSort_1024 is called from the CUDA categorical split-finder
    (cuda_best_split_finder.cu) over per-category gradient/hessian ratios.
    Training a small regression with a single categorical feature whose
    per-category sums are all-distinct exercises that path with non-tied
    scores; if the comparator stops alternating, CUDA's chosen categorical
    split disagrees with CPU's.
    """
    rng = np.random.default_rng(123)
    n = 400
    n_categories = 12
    cats = rng.integers(0, n_categories, size=n).astype(np.float64)
    # Per-category mean shift produces distinct, well-separated grad/hess
    # sums after fitting -- so the categorical sort sees no ties.
    category_means = rng.standard_normal(n_categories) * 0.7
    y = (category_means[cats.astype(int)] + rng.standard_normal(n) * 0.05).astype(np.float64)
    X = cats.reshape(-1, 1)

    base = {
        "objective": "regression",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
    }
    preds = {}
    for dev in ("cpu", "cuda"):
        ds = lgb.Dataset(
            X, label=y, categorical_feature=[0],
            params={"verbose": -1, "feature_pre_filter": False},
        )
        bst = lgb.train({**base, "device_type": dev}, ds, num_boost_round=1)
        preds[dev] = bst.predict(X, raw_score=True)
    diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    # If the bitonic sort stops alternating direction, the categorical
    # split-finder chooses a different threshold and predictions diverge by
    # ~O(category mean magnitude). 1e-3 is well above CPU/CUDA FP drift on a
    # one-tree fit but well below any wrong-split signal.
    assert diff < 1e-3, (
        f"CPU vs CUDA prediction disagreement on categorical split: max|Δ|={diff:.4e}"
    )


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
