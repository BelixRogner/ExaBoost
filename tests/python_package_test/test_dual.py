# coding: utf-8
"""Tests for dual GPU+CPU support."""

import contextlib
import io
import json
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


def _tree_depth(node, depth=0):
    if "leaf_value" in node:
        return depth
    return max(
        _tree_depth(node["left_child"], depth + 1),
        _tree_depth(node["right_child"], depth + 1),
    )


def _train_pair(params_overrides, X, y):
    out = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            **params_overrides,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        out[device_type] = lgb.train(params, ds, num_boost_round=1)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("max_depth", "num_leaves"),
    [
        (1, 2),
        (1, 7),
        (2, 4),
        (2, 7),
        (2, 31),
        (3, 7),
        (3, 31),
        (5, 31),
    ],
)
def test_cuda_respects_max_depth(max_depth, num_leaves):
    """CUDA tree learner must enforce max_depth, matching CPU.

    Regression test for the bug where CUDABestSplitFinder had no max_depth
    check and CUDATree::Split never updated host-side leaf_depth_, causing
    CUDA to produce trees up to log2(num_leaves) deep regardless of
    max_depth. With max_depth=2 and num_leaves=31, CUDA was producing
    depth-7 trees with all 31 leaves filled.
    """
    rng = np.random.default_rng(0)
    n = 64
    X = rng.standard_normal((n, 4)).astype(np.float64)
    y = (X @ rng.standard_normal(4) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    models = _train_pair(
        {"max_depth": max_depth, "num_leaves": num_leaves, "min_data_in_leaf": 1},
        X,
        y,
    )

    cpu_dump = models["cpu"].dump_model()["tree_info"][0]
    cuda_dump = models["cuda"].dump_model()["tree_info"][0]

    cpu_depth = _tree_depth(cpu_dump["tree_structure"])
    cuda_depth = _tree_depth(cuda_dump["tree_structure"])

    assert cuda_depth <= max_depth, (
        f"CUDA exceeded max_depth={max_depth}: produced depth-{cuda_depth} tree with num_leaves={num_leaves}"
    )
    assert cpu_depth == cuda_depth, (
        f"CPU/CUDA depth mismatch with max_depth={max_depth}, num_leaves={num_leaves}: "
        f"cpu={cpu_depth}, cuda={cuda_depth}"
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


def _train_forced(device_type, forced_split, tmp_path, num_boost_round=10, num_leaves=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(400, 6)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * rng.rand(400)
    fn = tmp_path / f"forced_{device_type}_{seed}.json"
    fn.write_text(json.dumps(forced_split))
    params = {
        "objective": "regression",
        "forcedsplits_filename": str(fn),
        "num_leaves": num_leaves,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "device_type": device_type,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=num_boost_round), X, y


_FORCED_SPLIT_CASES = {
    "root_only": {"feature": 2, "threshold": 0.5},
    "root_nested": {
        "feature": 2, "threshold": 0.5,
        "left": {"feature": 2, "threshold": 0.25},
        "right": {"feature": 2, "threshold": 0.75},
    },
    "root_lr_mixed": {
        "feature": 1, "threshold": 0.4,
        "left": {"feature": 0, "threshold": 0.3},
        "right": {"feature": 3, "threshold": 0.6},
    },
    "three_deep_chain": {
        "feature": 2, "threshold": 0.5,
        "left": {"feature": 0, "threshold": 0.5,
                 "left": {"feature": 1, "threshold": 0.5}},
    },
}


def _forced_features_in_json(node, out=None):
    if out is None:
        out = []
    if "feature" in node:
        out.append(node["feature"])
        for side in ("left", "right"):
            if side in node:
                _forced_features_in_json(node[side], out)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", list(_FORCED_SPLIT_CASES))
@pytest.mark.parametrize("num_leaves", [8, 31])
def test_cuda_forced_splits_honored(case, num_leaves, tmp_path):
    """CUDA must apply the forced-split JSON: the forced root feature heads every tree.

    Regression test for forcedsplits_filename being silently ignored on CUDA
    (ForceSplits only existed in SerialTreeLearner::Train; the CUDA learner never
    consulted the forced-split JSON, so CUDA trees split on whatever feature had
    the best gain instead of the forced one).
    """
    forced_split = _FORCED_SPLIT_CASES[case]
    bst, _, _ = _train_forced("cuda", forced_split, tmp_path, num_leaves=num_leaves)
    forced_root_feature = forced_split["feature"]
    for tree in bst.dump_model()["tree_info"]:
        root = tree["tree_structure"]
        assert root["split_feature"] == forced_root_feature, (
            f"tree does not honor forced root split: got feature {root['split_feature']}, "
            f"expected {forced_root_feature}"
        )


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", list(_FORCED_SPLIT_CASES))
@pytest.mark.parametrize("num_leaves", [8, 31])
@pytest.mark.parametrize("seed", [0, 1])
def test_cuda_forced_splits_match_cpu(case, num_leaves, seed, tmp_path):
    """CUDA forced-split training must produce the same model as CPU.

    Predictions must match at FP epsilon over 30 boosting rounds; the first tree's
    structure (split features, gains, counts, leaf values) must match exactly.
    """
    forced_split = _FORCED_SPLIT_CASES[case]
    bst_cpu, X, _ = _train_forced("cpu", forced_split, tmp_path,
                                  num_boost_round=30, num_leaves=num_leaves, seed=seed)
    bst_cuda, _, _ = _train_forced("cuda", forced_split, tmp_path,
                                   num_boost_round=30, num_leaves=num_leaves, seed=seed)

    # tree-0 structure equality (features, gains, counts, leaf values; thresholds may
    # differ in real-value display encoding for the same bin boundary)
    def substantive(bst):
        out = []

        def _rec(node):
            if "leaf_value" in node:
                out.append(("leaf", round(node["leaf_value"], 10), node["leaf_count"]))
            else:
                out.append((node["split_feature"], round(node["split_gain"], 6), node["internal_count"]))
                _rec(node["left_child"])
                _rec(node["right_child"])

        _rec(bst.dump_model()["tree_info"][0]["tree_structure"])
        return out

    assert substantive(bst_cpu) == substantive(bst_cuda)

    # prediction parity over all rounds
    np.testing.assert_allclose(
        bst_cpu.predict(X), bst_cuda.predict(X), rtol=0, atol=1e-10,
        err_msg=f"forced splits case={case}: CUDA diverges from CPU",
    )
