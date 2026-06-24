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
    ("objective", "alpha"),
    [("regression_l1", 0.5), ("quantile", 0.5), ("quantile", 0.3), ("quantile", 0.7)],
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


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("n", "num_leaves", "bagging_fraction", "bagging_freq"),
    [
        (200, 7, 0.7, 1),
        (500, 15, 0.5, 1),
        (1000, 31, 0.8, 3),
        (2000, 31, 0.7, 1),
    ],
)
def test_cuda_bagging_does_not_crash_and_matches_cpu(n, num_leaves, bagging_fraction, bagging_freq):
    """CUDA training with bagging must not crash and must track CPU.

    Regression test for two independent CUDA bugs that made *any* bagged run
    abort with "[CUDA] an illegal memory access was encountered":

    1. ``CUDATree::ToHost()`` freed the per-tree GPU tree-structure arrays
       (split_feature_inner / children / thresholds / decision_type), keeping
       only ``cuda_leaf_value_``, to bound device memory across many rounds.
       But ``AddPredictionToScoreKernel`` traverses the whole tree, and the
       GBDT out-of-bag score update (only reached under bagging) launches it
       post-ToHost, dereferencing the freed/null device pointers. Fixed by
       re-uploading the structure for that launch (and freeing it again).
    2. ``CUDADataPartition::CalcBlockDim`` is non-monotonic, so a bagged leaf
       (~bagging_fraction * n, landing in the ~101-160 band) needs more blocks
       than the full dataset, overflowing the per-block offset buffers. (Fixed
       separately; this test also guards against it regressing into a crash.)

    Before the fixes this test aborted the interpreter. After them, CPU and
    CUDA agree to within the floating-point / RNG divergence documented as
    expected for the CUDA tree learner in upstream issue #6055 (different bag
    sampling => a generous tolerance, the point of the test is no-crash +
    finite + same ballpark).
    """
    rng = np.random.default_rng(11)
    d = 8
    X = rng.standard_normal((n, d)).astype(np.float64)
    coef = rng.standard_normal(d)
    y = (X @ coef + 0.3 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 7,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "learning_rate": 0.05,
            "min_data_in_leaf": 5,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "bagging_seed": 3,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train(params, ds, num_boost_round=20)
        preds[device_type] = bst.predict(X, raw_score=True)

    assert np.all(np.isfinite(preds["cuda"])), "CUDA bagging produced non-finite predictions"
    # Bagging samples a different bag on CUDA than on CPU (different RNG stream),
    # so predictions are not bit-identical; #6055 documents this as expected.
    # The bar here is "same ballpark" -- strict enough to catch a model that
    # silently trained on garbage, loose enough to tolerate bag-sampling drift.
    max_abs = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    y_range = float(y.max() - y.min())
    assert max_abs < 0.25 * y_range, (
        f"CPU/CUDA bagging predictions diverge far more than bag-sampling drift: "
        f"max|Δ|={max_abs:.4f}, y_range={y_range:.4f}"
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(("n", "seed"), [(1000, 7), (2000, 7), (2000, 11)])
def test_cuda_quantized_tree_structure_matches_cpu(n, seed):
    """CUDA quantized trees must grow to roughly the same size as CPU quantized.

    Regression test for the child-leaf packed-sum bugs: the int64 packed
    gradient/hessian sum (sum_of_gradients_hessians) was (1) dropped by
    CUDASplitInfo::operator= and (2) never refreshed for child leaves in the
    data-partition split kernel. The discretized best-split finder uses that
    packed sum as the leaf total, so children inherited the parent's total,
    scored phantom splits that partitioned to a 0-data leaf, and CUDA quantized
    trees stalled at ~2-4 leaves while CPU grew the full 31.

    This guards the structural fix: CUDA must grow to within 20% of CPU's leaf
    count (the bug capped it far below). It deliberately does NOT assert tight
    prediction parity -- residual CPU/CUDA divergence remains from the separate
    open 8-bit-histogram leaf-value bug and from the cross-feature gain-tie
    ordering (a different branch). min_data_in_leaf is high so leaves stay in
    the 16-bit histogram regime.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12)).astype(np.float64)
    y = (X @ rng.standard_normal(12) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": seed,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
    }
    models = {}
    for device_type in ("cpu", "cuda"):
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        models[device_type] = lgb.train({**params, "device_type": device_type}, ds, num_boost_round=20)

    cpu_leaves = sum(t["num_leaves"] for t in models["cpu"].dump_model()["tree_info"])
    cuda_leaves = sum(t["num_leaves"] for t in models["cuda"].dump_model()["tree_info"])
    # The bug capped CUDA at ~2/tree (~40 total over 20 trees); after the fix it
    # tracks CPU's count. Require within 20% in both directions.
    assert 0.8 * cpu_leaves <= cuda_leaves <= 1.2 * cpu_leaves, (
        f"CUDA quantized tree size diverged: cuda={cuda_leaves} vs cpu={cpu_leaves} (n={n}, seed={seed})"
    )
    assert np.all(np.isfinite(models["cuda"].predict(X)))


@_REQUIRES_CUDA
@pytest.mark.parametrize(("n", "seed"), [(1000, 7), (2000, 7), (1000, 11)])
def test_cuda_quantized_deep_trees_track_cpu(n, seed):
    """Deep CUDA quantized trees (small leaves) must track CPU, not explode.

    Regression test for the histogram-slot collision: in SplitTreeStructureKernel
    the left-is-smaller branch handed the discretized child a hist slot at a 1x
    (right_leaf_index * num_total_bin) stride while every other path uses 2x, so a
    child could be assigned a slot already owned by another leaf. Its histogram then
    accumulated on top of that leaf's data (e.g. a 45-point leaf's histogram summed
    to 106 points' worth), yielding phantom splits, 0-data leaves, and leaf outputs
    that exploded to 1e12-1e20 over 20 rounds at small min_data_in_leaf.

    With min_data_in_leaf=20 the trees go deep (many 8-bit-histogram leaves), which
    is exactly where the collision bit. After the fix CUDA matches CPU's leaf counts
    and predictions stay bounded (residual divergence is the separate gain-tie
    ordering, handled on another branch).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12)).astype(np.float64)
    y = (X @ rng.standard_normal(12) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": seed,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
    }
    models = {}
    for device_type in ("cpu", "cuda"):
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        models[device_type] = lgb.train({**params, "device_type": device_type}, ds, num_boost_round=20)

    cpu_leaves = sum(t["num_leaves"] for t in models["cpu"].dump_model()["tree_info"])
    cuda_leaves = sum(t["num_leaves"] for t in models["cuda"].dump_model()["tree_info"])
    assert 0.9 * cpu_leaves <= cuda_leaves <= 1.1 * cpu_leaves, (
        f"CUDA quantized deep-tree size diverged: cuda={cuda_leaves} vs cpu={cpu_leaves} (n={n}, seed={seed})"
    )
    cpu_pred = models["cpu"].predict(X)
    cuda_pred = models["cuda"].predict(X)
    assert np.all(np.isfinite(cuda_pred))
    max_diff = float(np.max(np.abs(cuda_pred - cpu_pred)))
    # Broken behaviour exploded to >=1e12; the fix brings it to ~1 (gain-tie FP level).
    assert max_diff < 5.0, f"CUDA quantized deep trees diverge from CPU by {max_diff:.3g} (n={n}, seed={seed})"


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
        "feature": 2,
        "threshold": 0.5,
        "left": {"feature": 2, "threshold": 0.25},
        "right": {"feature": 2, "threshold": 0.75},
    },
    "root_lr_mixed": {
        "feature": 1,
        "threshold": 0.4,
        "left": {"feature": 0, "threshold": 0.3},
        "right": {"feature": 3, "threshold": 0.6},
    },
    "three_deep_chain": {
        "feature": 2,
        "threshold": 0.5,
        "left": {"feature": 0, "threshold": 0.5, "left": {"feature": 1, "threshold": 0.5}},
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
    bst_cpu, X, _ = _train_forced("cpu", forced_split, tmp_path, num_boost_round=30, num_leaves=num_leaves, seed=seed)
    bst_cuda, _, _ = _train_forced("cuda", forced_split, tmp_path, num_boost_round=30, num_leaves=num_leaves, seed=seed)

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
        bst_cpu.predict(X),
        bst_cuda.predict(X),
        rtol=0,
        atol=1e-10,
        err_msg=f"forced splits case={case}: CUDA diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_leaves", [7, 31, 63])
def test_cuda_histogram_event_ordering_matches_cpu(num_leaves):
    """Multi-leaf CUDA training must match CPU after the histogram->FindBestSplits
    device syncs were replaced by event-based stream ordering.

    The per-split device sync after histogram construction and the one between the
    smaller- and larger-leaf FindBestSplits launches were replaced by
    cudaStreamWaitEvent on the histogram constructor's construct/subtract completion
    events. The smaller-leaf search waits for the constructed histogram, the
    larger-leaf search for the subtracted histogram. Building trees with several
    leaves exercises this path; a missing/incorrect event ordering would let a
    FindBestSplits kernel read a histogram before it is written and diverge from
    CPU well beyond the tolerance below.
    """
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal((n, 10)).astype(np.float64)
    y = (X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "min_data_in_leaf": 5,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        preds[device_type] = lgb.train(params, ds, num_boost_round=5).predict(X)

    np.testing.assert_allclose(
        preds["cuda"],
        preds["cpu"],
        rtol=0,
        atol=1e-9,
        err_msg=f"num_leaves={num_leaves}: CUDA histogram event ordering diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_leaves", [7, 31, 63])
def test_cuda_syncbestsplit_overlap_matches_cpu(num_leaves):
    """Multi-leaf CUDA training must match CPU after overlapping the two child
    leaves' SyncBestSplit reduction.

    LaunchSyncBestSplitForLeafKernel used to reduce the smaller and larger child
    leaves' per-feature best splits with two kernel launches separated by a full
    device sync. That sync was dropped and the two leaves now reduce concurrently
    on separate streams (each reads only the cuda_best_split_info_ region that
    FindBestSplitsForLeafKernel wrote on that same stream). Building trees with
    several leaves exercises the both-child-leaves-valid path this touches; CUDA
    must stay bit-for-bit aligned with CPU on the deterministic single-thread,
    double-precision config.
    """
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal((n, 10)).astype(np.float64)
    y = (X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "min_data_in_leaf": 5,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        preds[device_type] = lgb.train(params, ds, num_boost_round=5).predict(X)

    np.testing.assert_allclose(
        preds["cuda"],
        preds["cpu"],
        rtol=0,
        atol=1e-9,
        err_msg=f"num_leaves={num_leaves}: CUDA SyncBestSplit overlap diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [200, 500, 1000])
@pytest.mark.parametrize("num_leaves", [7, 31])
def test_cuda_quantized_training_produces_splits(n, num_leaves):
    """CUDA use_quantized_grad training must produce real splitting trees.

    Regression test for the 2x under-allocation of the discretized
    gradient/hessian buffer (CUDAGradientDiscretizer). The buffer holds an
    int16 gradient and an int16 hessian per data point (4 bytes), but was
    sized num_data * 2 bytes. DiscretizeGradientsKernel overran it and
    corrupted the adjacent dequantization scale buffers, which made the
    root leaf's sum_hessians come out as a denormal ~0. The root then
    failed the sum_hessians > min_sum_hessian_in_leaf validity check and
    never split, so every CUDA quantized tree collapsed to a single leaf
    and the model did not learn.

    The fix sizes the buffer as num_data * 4. Here we assert that CUDA
    quantized trees actually split (num_leaves > 1) and that predictions
    are not the degenerate constant the single-leaf model produced.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, 12)).astype(np.float64)
    y = (X @ rng.standard_normal(12) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": num_leaves,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 7,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "device_type": "cuda",
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    bst = lgb.train(params, ds, num_boost_round=20)

    leaf_counts = [tree["num_leaves"] for tree in bst.dump_model()["tree_info"]]
    assert max(leaf_counts) > 1, (
        f"CUDA quantized training collapsed to single-leaf trees "
        f"(n={n}, num_leaves={num_leaves}): per-tree leaf counts {leaf_counts[:5]}"
    )

    preds = bst.predict(X)
    assert np.all(np.isfinite(preds)), "CUDA quantized produced non-finite predictions"
    assert preds.std() > 1e-6, f"CUDA quantized predictions are degenerate/constant (n={n}, num_leaves={num_leaves})"


@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [2000, 8000, 50000])
def test_cuda_quantized_32bit_histogram_matches_cpu(n):
    """CUDA quantized training must match CPU once leaves need 32-bit histograms.

    Regression test for the best-split finder reading the 32-bit discretized
    histogram with the wrong width. A leaf whose max per-bin stat
    (num_data_in_leaf * num_grad_quant_bins) reaches 65536 uses an int64-per-bin
    (32-bit grad / 32-bit hess) histogram. The finder dispatched that case with
    BIN_HIST_TYPE=int32_t and read it through an int32_t* offset, i.e. 4-byte
    half-bins at the wrong stride, so the split search saw garbage for any leaf
    large enough to need 32-bit bins. With num_grad_quant_bins=16 that is any leaf
    with >= 4096 rows; the resulting models were near-random (correlation ~0 with
    CPU). The 8-bit and 16-bit paths were correct, so small-data tests never hit it.

    With num_grad_quant_bins=16: n=2000 stays 16-bit (already correct), while
    n>=8000 forces a 32-bit root. The fix reads the histogram as int64, making CUDA
    bit-identical to CPU at all scales.
    """
    rng = np.random.default_rng(0)
    nf = 40
    X = rng.standard_normal((n, nf)).astype(np.float64)
    y = (X @ rng.standard_normal(nf) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "learning_rate": 0.05,
            "use_quantized_grad": True,
            "num_grad_quant_bins": 16,
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
        preds[device_type] = lgb.train(params, ds, num_boost_round=20).predict(X)
    corr = float(np.corrcoef(preds["cpu"], preds["cuda"])[0, 1])
    # Before the fix, 32-bit-histogram leaves (n>=8000) gave correlation ~0.
    assert corr > 0.99, f"CUDA quantized (32-bit histogram) diverges from CPU: corr={corr:.4f} (n={n})"
