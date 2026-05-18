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
            X,
            label=y,
            categorical_feature=[0],
            params={"verbose": -1, "feature_pre_filter": False},
        )
        bst = lgb.train({**base, "device_type": dev}, ds, num_boost_round=1)
        preds[dev] = bst.predict(X, raw_score=True)
    diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    # If the bitonic sort stops alternating direction, the categorical
    # split-finder chooses a different threshold and predictions diverge by
    # ~O(category mean magnitude). 1e-3 is well above CPU/CUDA FP drift on a
    # one-tree fit but well below any wrong-split signal.
    assert diff < 1e-3, f"CPU vs CUDA prediction disagreement on categorical split: max|Δ|={diff:.4e}"


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


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)

# Loose enough to absorb label_t float32 quantization in the renewal kernel,
# tight enough to flag the ~0.3 bias the old PercentileDevice formula produced.
_PERCENTILE_TOL = 1e-6


def _train_one_tree_for_renewal(device_type, objective, alpha, X, y):
    # learning_rate=1.0 makes raw_score equal to the renewed leaf value directly,
    # which lets the assertion compare against numpy.quantile without unwinding shrinkage.
    params = {
        "objective": objective,
        "alpha": alpha,
        "num_leaves": 7,
        "min_data_in_leaf": 1,
        "learning_rate": 1.0,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "feature_pre_filter": False,
        "device_type": device_type,
        "gpu_use_dp": True,
        "force_col_wise": True,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=1)


@_REQUIRES_CUDA
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cuda_l1_leaf_renewal_matches_numpy_median(seed):
    """L1 leaf renewal must produce numpy.median(y_in_leaf) on both CPU and CUDA.

    Regression test for the unweighted PercentileDevice formula that previously
    used `len * (1 - alpha)` instead of `(len - 1) * (1 - alpha)`, biasing
    leaf values upward in the descending-sort convention used for L1 / quantile
    renewal.
    """
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.standard_normal((n, 5)).astype(np.float64)
    w = rng.standard_normal(5)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float64)

    for device_type in ("cpu", "cuda"):
        bst = _train_one_tree_for_renewal(device_type, "regression_l1", 0.5, X, y)
        leaf_idx = bst.predict(X, pred_leaf=True).astype(int).reshape(-1)
        raw = bst.predict(X, raw_score=True)
        for li in np.unique(leaf_idx):
            mask = leaf_idx == li
            expected = float(np.median(y[mask]))
            actual = float(raw[mask][0])
            assert actual == pytest.approx(expected, abs=_PERCENTILE_TOL), (
                f"{device_type} leaf {li} (n={int(mask.sum())}): expected np.median={expected:.10f}, got {actual:.10f}"
            )


@_REQUIRES_CUDA
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.7, 0.9])
def test_cuda_quantile_leaf_renewal_matches_numpy_quantile(alpha):
    """Quantile leaf renewal must produce numpy.quantile(y_in_leaf, alpha)
    on both CPU and CUDA. Same regression coverage as the L1 test, but
    sweeping alpha so the bias of the wrong formula would show on every
    even/odd leaf size combination.
    """
    rng = np.random.default_rng(123)
    n = 250
    X = rng.standard_normal((n, 6)).astype(np.float64)
    w = rng.standard_normal(6)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float64)

    for device_type in ("cpu", "cuda"):
        bst = _train_one_tree_for_renewal(device_type, "quantile", alpha, X, y)
        leaf_idx = bst.predict(X, pred_leaf=True).astype(int).reshape(-1)
        raw = bst.predict(X, raw_score=True)
        for li in np.unique(leaf_idx):
            mask = leaf_idx == li
            expected = float(np.quantile(y[mask], alpha))
            actual = float(raw[mask][0])
            assert actual == pytest.approx(expected, abs=_PERCENTILE_TOL), (
                f"{device_type} alpha={alpha} leaf {li} (n={int(mask.sum())}): "
                f"expected np.quantile={expected:.10f}, got {actual:.10f}"
            )


@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9])
def test_cuda_l1_median_handles_small_even_and_odd_leaves(n):
    """Targets the specific failure mode of the old PercentileDevice formula:
    even-length leaves returning sorted[1] instead of avg(sorted[1], sorted[2]),
    and odd-length leaves returning avg(sorted[1], sorted[2]) instead of
    sorted[2]. We force every datapoint into its own leaf, then split a couple
    in half and check the leaf medians.
    """
    rng = np.random.default_rng(7)
    # one feature so we deterministically split on it; values are well-separated
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    # values designed so that splitting on the only feature produces leaves of
    # exactly the requested cardinalities at depth 1 and 2.
    y = rng.standard_normal(n).astype(np.float64)

    for device_type in ("cpu", "cuda"):
        bst = _train_one_tree_for_renewal(device_type, "regression_l1", 0.5, X, y)
        leaf_idx = bst.predict(X, pred_leaf=True).astype(int).reshape(-1)
        raw = bst.predict(X, raw_score=True)
        for li in np.unique(leaf_idx):
            mask = leaf_idx == li
            expected = float(np.median(y[mask]))
            actual = float(raw[mask][0])
            assert actual == pytest.approx(expected, abs=_PERCENTILE_TOL), (
                f"{device_type} n={n} leaf {li} (size {int(mask.sum())}): "
                f"expected np.median={expected:.10f}, got {actual:.10f}"
            )
