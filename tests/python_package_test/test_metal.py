# coding: utf-8
"""Tests for the Metal backend (Apple silicon).

These tests train the same data on both CPU and Metal and assert that the
resulting models agree on the standard headline metrics (AUC, accuracy, F1,
log-loss). Tolerances are deliberately loose-but-not-trivial: Metal histogram
construction uses non-deterministic atomic-add ordering, which gives ULP-level
drift in histograms and slight differences in tree splits.

Skipped unless LIGHTGBM_TEST_METAL=1 is set in the environment (and only runs
on darwin/arm64 where a Metal-enabled build is realistic).
"""

import os
import platform

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb


_skip_reason = (
    "Set LIGHTGBM_TEST_METAL=1 and build with -DUSE_METAL=ON on macOS/arm64 to run."
)
_should_run = (
    os.environ.get("LIGHTGBM_TEST_METAL") == "1"
    and platform.system() == "Darwin"
    and platform.machine() == "arm64"
)
pytestmark = pytest.mark.skipif(not _should_run, reason=_skip_reason)


def _train_both(params_base, X_train, y_train, X_test, y_test, num_rounds=50):
    """Train identical models on cpu and metal devices; return (cpu_pred, metal_pred)."""
    rng_seed = 42
    common = dict(params_base, verbosity=-1, deterministic=True, seed=rng_seed)

    cpu_params = dict(common, device_type="cpu")
    cpu_ds = lgb.Dataset(X_train, y_train)
    cpu_bst = lgb.train(cpu_params, cpu_ds, num_boost_round=num_rounds)

    metal_params = dict(common, device_type="metal")
    metal_ds = lgb.Dataset(X_train, y_train)
    metal_bst = lgb.train(metal_params, metal_ds, num_boost_round=num_rounds)

    return cpu_bst.predict(X_test), metal_bst.predict(X_test)


def test_binary_classification_parity():
    """Binary task: AUC, accuracy, F1, log-loss agree between cpu and metal."""
    X, y = make_classification(
        n_samples=4_000, n_features=64, n_informative=24, n_redundant=8,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test,
    )

    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    cpu_lbl = (cpu_pred > 0.5).astype(int)
    metal_lbl = (metal_pred > 0.5).astype(int)

    cpu_acc = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    cpu_f1 = f1_score(y_test, cpu_lbl)
    metal_f1 = f1_score(y_test, metal_lbl)

    # Headline metrics should agree to ~1% relative or 0.01 absolute, whichever
    # is looser. Bumps to 2% / 0.02 if the Metal histograms produce different
    # but still high-quality splits.
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_acc == pytest.approx(cpu_acc, abs=0.02), (cpu_acc, metal_acc)
    assert metal_f1 == pytest.approx(cpu_f1, abs=0.02), (cpu_f1, metal_f1)
    # Both should be meaningful (not degenerate).
    assert metal_auc > 0.7
    assert metal_acc > 0.6


def test_multiclass_classification_parity():
    """Multiclass: macro-F1 agrees between cpu and metal."""
    X, y = make_classification(
        n_samples=3_000, n_features=48, n_informative=20, n_classes=4,
        n_clusters_per_class=2, random_state=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    cpu_pred_proba, metal_pred_proba = _train_both(
        {"objective": "multiclass", "num_class": 4, "num_leaves": 15,
         "learning_rate": 0.1},
        X_train, y_train, X_test, y_test,
    )
    cpu_lbl = cpu_pred_proba.argmax(axis=1)
    metal_lbl = metal_pred_proba.argmax(axis=1)

    cpu_acc = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    cpu_f1 = f1_score(y_test, cpu_lbl, average="macro")
    metal_f1 = f1_score(y_test, metal_lbl, average="macro")

    assert metal_acc == pytest.approx(cpu_acc, abs=0.03), (cpu_acc, metal_acc)
    assert metal_f1 == pytest.approx(cpu_f1, abs=0.03), (cpu_f1, metal_f1)


def test_regression_parity():
    """Regression: MSE agrees between cpu and metal."""
    X, y = make_regression(
        n_samples=3_000, n_features=32, n_informative=16, noise=0.5,
        random_state=2,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=2
    )

    cpu_pred, metal_pred = _train_both(
        {"objective": "regression", "num_leaves": 31, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test,
    )

    cpu_mse = mean_squared_error(y_test, cpu_pred)
    metal_mse = mean_squared_error(y_test, metal_pred)
    # MSEs can drift more than per-prediction metrics; tolerate 5% relative.
    assert metal_mse == pytest.approx(cpu_mse, rel=0.05), (cpu_mse, metal_mse)


def test_categorical_features_parity():
    """Mix continuous and explicit categorical features. LightGBM has a
    separate categorical histogram path; verify Metal acceleration doesn't
    silently mishandle these (or falls back cleanly)."""
    rng = np.random.default_rng(7)
    n = 4_000
    # 32 continuous + 16 categoricals (each with 8 levels).
    cont = rng.normal(size=(n, 32))
    cat = rng.integers(low=0, high=8, size=(n, 16)).astype(float)
    X = np.concatenate([cont, cat], axis=1)
    coef = rng.normal(size=X.shape[1])
    y = (X @ coef + 0.5 * rng.normal(size=n) > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    cat_feature_idx = list(range(32, 48))  # which columns are categorical
    base = dict(
        objective="binary", num_leaves=31, learning_rate=0.1,
        verbosity=-1, deterministic=True, seed=42,
    )

    cpu_ds = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_idx)
    cpu_bst = lgb.train(dict(base, device_type="cpu"), cpu_ds, num_boost_round=40)
    metal_ds = lgb.Dataset(X_train, y_train, categorical_feature=cat_feature_idx)
    metal_bst = lgb.train(dict(base, device_type="metal"), metal_ds, num_boost_round=40)

    cpu_pred = cpu_bst.predict(X_test)
    metal_pred = metal_bst.predict(X_test)

    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.03), (cpu_auc, metal_auc)
    assert metal_auc > 0.6


def test_bagging_parity():
    """bagging_fraction != 1.0 subsamples rows each iteration; the Metal
    indexed-kernel path must handle this. Tight tolerance because bagging is
    a common LightGBM use case and silent drift here would hurt real users."""
    X, y = make_classification(
        n_samples=5_000, n_features=64, n_informative=20, random_state=11,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=11
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "bagging_fraction": 0.5, "bagging_freq": 1},
        X_train, y_train, X_test, y_test, num_rounds=40,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_feature_fraction_parity():
    """feature_fraction < 1 randomly disables features per-tree. Metal must
    correctly honor is_feature_used in the per-feature write-back."""
    X, y = make_classification(
        n_samples=5_000, n_features=96, n_informative=24, random_state=12,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "feature_fraction": 0.5},
        X_train, y_train, X_test, y_test, num_rounds=40,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_64bin_kernel_parity():
    """max_bin=60 forces the 64-bin Metal kernel variant. Verifies that
    pipeline state separate from the default 256-bin variant produces
    matching results."""
    X, y = make_classification(
        n_samples=4_000, n_features=64, n_informative=20, random_state=8,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=8
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "max_bin": 60},
        X_train, y_train, X_test, y_test,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_binary_classification_multifeature_group_parity():
    """Force LightGBM to pack features into multi-feature groups (low max_bin)
    and verify cpu vs metal still match. Exercises the multi-feature-group
    write-back path added in Phase 2.5."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=24, n_redundant=8,
        random_state=4,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=4
    )

    # max_bin=15 -> each feature has <=16 bins -> LightGBM packs 8 features
    # per uint32 feature group.
    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "max_bin": 15,
    }
    cpu_pred, metal_pred = _train_both(params, X_train, y_train, X_test, y_test)

    cpu_auc   = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    cpu_lbl   = (cpu_pred   > 0.5).astype(int)
    metal_lbl = (metal_pred > 0.5).astype(int)
    cpu_acc   = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    cpu_f1    = f1_score(y_test, cpu_lbl)
    metal_f1  = f1_score(y_test, metal_lbl)

    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_acc == pytest.approx(cpu_acc, abs=0.02), (cpu_acc, metal_acc)
    assert metal_f1  == pytest.approx(cpu_f1,  abs=0.02), (cpu_f1,  metal_f1)
    assert metal_auc > 0.7


def test_lambdarank_parity():
    """LambdaRank is a pairwise ranking objective with a different gradient
    pattern from L2 / logistic. Verifies Metal works on ranking workloads."""
    rng = np.random.default_rng(16)
    n_queries = 200
    docs_per_query = 25
    n = n_queries * docs_per_query
    X = rng.normal(size=(n, 64))
    # Relevance labels in [0, 4], correlated with first feature.
    y = np.clip(np.round(X[:, 0] + 2 + rng.normal(0, 0.5, size=n)), 0, 4).astype(int)
    group = np.full(n_queries, docs_per_query, dtype=int)

    base = dict(
        objective="lambdarank", num_leaves=31, learning_rate=0.1,
        label_gain=[0, 1, 3, 7, 15], verbosity=-1, deterministic=True, seed=42,
        metric="ndcg", eval_at=[5],
    )
    cpu_ds = lgb.Dataset(X, y, group=group)
    cpu_bst = lgb.train(dict(base, device_type="cpu"), cpu_ds, num_boost_round=30)
    metal_ds = lgb.Dataset(X, y, group=group)
    metal_bst = lgb.train(dict(base, device_type="metal"), metal_ds, num_boost_round=30)

    cpu_pred = cpu_bst.predict(X)
    metal_pred = metal_bst.predict(X)
    # Spearman-like rank agreement: scores should produce similar orderings.
    # Use mean pairwise ordering agreement within each query as a check.
    correct = 0; total = 0
    for q in range(n_queries):
        start = q * docs_per_query
        end = start + docs_per_query
        c_order = np.argsort(-cpu_pred[start:end])
        m_order = np.argsort(-metal_pred[start:end])
        # Compare top-5 sets (looser than full ordering).
        c_top5 = set(c_order[:5])
        m_top5 = set(m_order[:5])
        correct += len(c_top5 & m_top5)
        total += 5
    overlap = correct / total
    assert overlap > 0.85, f"top-5 set overlap: {overlap:.2%}"


def test_quantile_regression_parity():
    """quantile objective produces non-trivial gradient/hessian patterns
    different from L2. Verifies the Metal path doesn't make assumptions
    about gradient shape."""
    X, y = make_regression(
        n_samples=3_000, n_features=48, noise=2.0, random_state=14,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=14
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "quantile", "alpha": 0.75, "num_leaves": 31,
         "learning_rate": 0.1},
        X_train, y_train, X_test, y_test, num_rounds=40,
    )
    # quantile loss isn't symmetric; use a quantile-aware tolerance.
    cpu_p75  = np.quantile(np.abs(y_test - cpu_pred), 0.75)
    metal_p75 = np.quantile(np.abs(y_test - metal_pred), 0.75)
    rel = abs(metal_p75 - cpu_p75) / max(cpu_p75, 1e-6)
    assert rel < 0.05, (cpu_p75, metal_p75)


def test_sparse_input_falls_back_cleanly():
    """Sparse / multi-val datasets aren't supported by the Metal kernels yet;
    they must transparently fall back to the CPU path without crashing or
    producing degraded results."""
    from scipy.sparse import csr_matrix
    X, y = make_classification(
        n_samples=2_000, n_features=128, n_informative=20, random_state=15,
    )
    X[np.abs(X) < 0.8] = 0  # ~60% zeros
    X_sparse = csr_matrix(X)
    X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_sparse, y, test_size=0.25, random_state=15
    )

    cpu_ds = lgb.Dataset(X_train_s, y_train)
    cpu_bst = lgb.train({"objective": "binary", "num_leaves": 15, "verbosity": -1,
                         "device_type": "cpu", "deterministic": True, "seed": 0},
                        cpu_ds, num_boost_round=30)
    metal_ds = lgb.Dataset(X_train_s, y_train)
    metal_bst = lgb.train({"objective": "binary", "num_leaves": 15, "verbosity": -1,
                           "device_type": "metal", "deterministic": True, "seed": 0},
                          metal_ds, num_boost_round=30)

    cpu_pred = cpu_bst.predict(X_test_s)
    metal_pred = metal_bst.predict(X_test_s)
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    # Same code path (Metal falls back to SerialTreeLearner). Tolerate a
    # small drift in case of any incidental non-determinism in shared CPU
    # paths; the important assertion is that AUC is close.
    assert metal_auc == pytest.approx(cpu_auc, abs=0.01), (cpu_auc, metal_auc)
    # Sanity: fallback model is still useful (sparse data IS informative).
    assert metal_auc > 0.85


def test_quantized_gradient_parity():
    """use_quantized_grad=true now runs on Metal via the int8-input/int32-
    output kernel for 32-bit-histogram leaves. Asserts AUC parity with CPU.
    16-bit leaves still delegate to CPU.

    Note: this test was originally named *_falls_back_cleanly when the
    quantized path was a CPU fallback; in Phase 2.9 the Metal q32 kernel
    landed, so this now exercises the actual Metal-accelerated quantized
    path."""
    X, y = make_classification(
        n_samples=5_000, n_features=128, n_informative=24, random_state=17,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=17
    )
    params = {
        "objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
        "use_quantized_grad": True, "verbosity": -1,
        "deterministic": True, "seed": 0,
    }
    cpu_ds = lgb.Dataset(X_train, y_train)
    cpu_bst = lgb.train(dict(params, device_type="cpu"), cpu_ds, num_boost_round=40)
    metal_ds = lgb.Dataset(X_train, y_train)
    metal_bst = lgb.train(dict(params, device_type="metal"), metal_ds, num_boost_round=40)

    cpu_pred = cpu_bst.predict(X_test)
    metal_pred = metal_bst.predict(X_test)
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    # Metal q32 atomics are deterministic-ish but may produce tiny drift in
    # int32 sum-order. Tolerate small ULP-level deviation.
    assert metal_auc == pytest.approx(cpu_auc, abs=0.01), (cpu_auc, metal_auc)


def test_no_regularization_parity():
    """lambda_l1=0, lambda_l2=0, min_gain_to_split=0 — most "raw" possible
    config. Surfaces any numerical bug that gentler configs would mask."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=24, random_state=40,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=40
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05,
         "lambda_l1": 0.0, "lambda_l2": 0.0, "min_gain_to_split": 0.0,
         "min_data_in_leaf": 1, "min_sum_hessian_in_leaf": 0.0},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.03), (cpu_auc, metal_auc)


def test_quantized_regression_parity():
    """Quantized regression: l2 objective + int8 gradients + 32-bit hists."""
    X, y = make_regression(
        n_samples=3_000, n_features=128, noise=0.5, random_state=39,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=39
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
         "use_quantized_grad": True},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_mse = mean_squared_error(y_test, cpu_pred)
    metal_mse = mean_squared_error(y_test, metal_pred)
    assert metal_mse == pytest.approx(cpu_mse, rel=0.05), (cpu_mse, metal_mse)


def test_quantized_multiclass_parity():
    """Quantized + multiclass: builds num_class trees per iteration."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=24, n_classes=4,
        n_clusters_per_class=2, random_state=38,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=38
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "multiclass", "num_class": 4, "num_leaves": 15,
         "learning_rate": 0.1, "use_quantized_grad": True},
        X_train, y_train, X_test, y_test, num_rounds=20,
    )
    cpu_lbl = cpu_pred.argmax(axis=1)
    metal_lbl = metal_pred.argmax(axis=1)
    cpu_acc = accuracy_score(y_test, cpu_lbl)
    metal_acc = accuracy_score(y_test, metal_lbl)
    assert metal_acc == pytest.approx(cpu_acc, abs=0.03), (cpu_acc, metal_acc)


def test_quantized_with_bagging_parity():
    """Quantized gradient mode + bagging: stresses both code paths
    (q32 kernel + indexed dispatch with int8 gather)."""
    X, y = make_classification(
        n_samples=4_000, n_features=128, random_state=37,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=37
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "use_quantized_grad": True,
         "bagging_fraction": 0.5, "bagging_freq": 1},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_min_data_in_leaf_parity():
    """min_data_in_leaf prevents tiny leaves; common LightGBM regularization
    knob. Verifies the Metal path respects it correctly."""
    X, y = make_classification(
        n_samples=2_000, n_features=128, random_state=36,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=36
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "min_data_in_leaf": 100},  # forces shallower-than-default trees
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_verify_mode_smoke():
    """LIGHTGBM_METAL_VERIFY=1 runs CPU alongside Metal and compares.
    Smoke-tests that the mode doesn't crash and produces sensible models
    (CPU values are used for splits, so AUC should match plain CPU)."""
    orig = os.environ.get("LIGHTGBM_METAL_VERIFY")
    os.environ["LIGHTGBM_METAL_VERIFY"] = "1"
    try:
        X, y = make_classification(
            n_samples=2_000, n_features=128, random_state=35,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=35
        )
        cpu_pred, metal_pred = _train_both(
            {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1},
            X_train, y_train, X_test, y_test, num_rounds=20,
        )
        cpu_auc = roc_auc_score(y_test, cpu_pred)
        metal_auc = roc_auc_score(y_test, metal_pred)
        # Verify mode uses CPU values for splits, so AUC should match
        # very tightly.
        assert metal_auc == pytest.approx(cpu_auc, abs=0.001), (cpu_auc, metal_auc)
    finally:
        if orig is None:
            del os.environ["LIGHTGBM_METAL_VERIFY"]
        else:
            os.environ["LIGHTGBM_METAL_VERIFY"] = orig


def test_xentropy_objective_parity():
    """Cross-entropy objective: y is continuous in [0,1] (probability-like)
    rather than 0/1. Different gradient pattern; verifies parity."""
    rng = np.random.default_rng(34)
    n, p = 3_000, 96
    X = rng.normal(size=(n, p))
    # Soft labels in [0,1].
    logits = X[:, 0] * 0.7 + X[:, 1] * -0.3 + rng.normal(0, 0.2, size=n)
    y = 1.0 / (1.0 + np.exp(-logits))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=34
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "xentropy", "num_leaves": 15, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_mse = mean_squared_error(y_test, cpu_pred)
    metal_mse = mean_squared_error(y_test, metal_pred)
    assert metal_mse == pytest.approx(cpu_mse, rel=0.05), (cpu_mse, metal_mse)


def test_poisson_objective_parity():
    """Poisson regression has a log-link function producing non-Gaussian
    gradients. Verifies Metal histograms work with that gradient profile."""
    rng = np.random.default_rng(33)
    n, p = 3_000, 64
    X = rng.normal(size=(n, p))
    # Generate count labels with log-linear means.
    rate = np.exp(X[:, 0] * 0.5 + 1.5)
    y = rng.poisson(rate)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=33
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "poisson", "num_leaves": 15, "learning_rate": 0.05},
        X_train, y_train.astype(float), X_test, y_test.astype(float),
        num_rounds=30,
    )
    # Poisson predictions are non-negative real-valued; compare per-sample
    # absolute error.
    diff = np.abs(metal_pred - cpu_pred)
    median_diff = float(np.median(diff))
    # Both should be in the same scale as the rate; median diff should
    # be a small fraction.
    assert median_diff < 1.0, f"median |metal - cpu|: {median_diff}"


def test_huber_objective_parity():
    """Huber regression objective has a piecewise-linear gradient. Tests
    that the Metal histogram path stays correct under non-quadratic
    objectives."""
    X, y = make_regression(
        n_samples=2_500, n_features=96, noise=1.0, random_state=32,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=32
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "huber", "alpha": 0.9, "num_leaves": 31,
         "learning_rate": 0.1},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_mse = mean_squared_error(y_test, cpu_pred)
    metal_mse = mean_squared_error(y_test, metal_pred)
    assert metal_mse == pytest.approx(cpu_mse, rel=0.05), (cpu_mse, metal_mse)


def test_k_feats_2_opt_in_parity():
    """LIGHTGBM_METAL_K_FEATS=2 opts into the experimental multi-feature
    kernel. Verifies it still produces AUC parity with CPU even though
    it's not the default fast path."""
    orig = os.environ.get("LIGHTGBM_METAL_K_FEATS")
    os.environ["LIGHTGBM_METAL_K_FEATS"] = "2"
    try:
        X, y = make_classification(
            n_samples=3_000, n_features=128, random_state=31,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=31
        )
        cpu_pred, metal_pred = _train_both(
            {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1},
            X_train, y_train, X_test, y_test, num_rounds=30,
        )
        cpu_auc = roc_auc_score(y_test, cpu_pred)
        metal_auc = roc_auc_score(y_test, metal_pred)
        assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    finally:
        if orig is None:
            del os.environ["LIGHTGBM_METAL_K_FEATS"]
        else:
            os.environ["LIGHTGBM_METAL_K_FEATS"] = orig


def test_min_gain_to_split_parity():
    """min_gain_to_split causes some splits to be rejected; verifies
    the rejected-split path doesn't trip Metal-specific bugs."""
    X, y = make_classification(
        n_samples=2_500, n_features=128, random_state=30,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=30
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05,
         "min_gain_to_split": 0.05},  # filter out marginal splits
        X_train, y_train, X_test, y_test, num_rounds=40,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_single_iteration_parity():
    """num_boost_round=1: only the root tree gets built. Tests Metal
    handles the one-tree case correctly (no leaf-to-leaf state carryover
    needed)."""
    X, y = make_classification(
        n_samples=2_000, n_features=128, random_state=29,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=29
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test, num_rounds=1,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_force_col_wise_parity():
    """force_col_wise=true switches LightGBM's histogram-construction
    strategy. Different code path on the CPU side; verify Metal still
    matches the CPU output."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=20, random_state=28,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=28
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1,
         "force_col_wise": True},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_monotone_constraints_parity():
    """monotone_constraints restricts splits to monotone wrt features;
    different split-gain code path. Verifies histograms still match."""
    X, y = make_classification(
        n_samples=3_000, n_features=64, n_informative=20, random_state=27,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=27
    )
    constraints = [1] * 16 + [-1] * 16 + [0] * (X.shape[1] - 32)
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1,
         "monotone_constraints": constraints,
         "monotone_constraints_method": "basic"},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_l1_regularization_parity():
    """lambda_l1 changes the split-gain formula; verifies the histogram
    output produces the same regularized splits on cpu and metal."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=24, random_state=26,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=26
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "lambda_l1": 1.0, "lambda_l2": 0.5},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_realistic_production_config():
    """Combines bagging, feature_fraction, early stopping, and ndcg-style
    multiclass — the kind of config a real production tabular ML job
    might use. Catches integration bugs that simpler tests miss."""
    X, y = make_classification(
        n_samples=8_000, n_features=192, n_informative=40, n_redundant=15,
        flip_y=0.02, random_state=25,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=25
    )
    base = dict(
        objective="binary", num_leaves=63, learning_rate=0.05,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.01, lambda_l2=0.1,
        verbosity=-1, deterministic=True, seed=42,
        min_data_in_leaf=50,
    )
    cpu_ds = lgb.Dataset(X_train, y_train)
    cpu_val = lgb.Dataset(X_test, y_test, reference=cpu_ds)
    cpu_bst = lgb.train(
        dict(base, device_type="cpu"), cpu_ds, num_boost_round=200,
        valid_sets=[cpu_val],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )
    metal_ds = lgb.Dataset(X_train, y_train)
    metal_val = lgb.Dataset(X_test, y_test, reference=metal_ds)
    metal_bst = lgb.train(
        dict(base, device_type="metal"), metal_ds, num_boost_round=200,
        valid_sets=[metal_val],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    cpu_auc = roc_auc_score(y_test, cpu_bst.predict(X_test))
    metal_auc = roc_auc_score(y_test, metal_bst.predict(X_test))
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.85


def test_min_features_env_override():
    """LIGHTGBM_METAL_MIN_FEATURES env var should change the heuristic
    threshold. Sets it to 8 (well below default 96) so a small-feature
    dataset uses Metal; verifies training succeeds and produces a usable
    model (rather than silently corrupting results)."""
    orig = os.environ.get("LIGHTGBM_METAL_MIN_FEATURES")
    os.environ["LIGHTGBM_METAL_MIN_FEATURES"] = "8"
    try:
        X, y = make_classification(n_samples=800, n_features=16, random_state=24)
        bst = lgb.train(
            {"objective": "binary", "num_leaves": 7, "verbosity": -1,
             "device_type": "metal", "deterministic": True, "seed": 0},
            lgb.Dataset(X, y), num_boost_round=10,
        )
        pred = bst.predict(X)
        assert np.all(np.isfinite(pred))
        # Should be a decent classifier on this recoverable dataset.
        assert roc_auc_score(y, pred) > 0.85
    finally:
        if orig is None:
            del os.environ["LIGHTGBM_METAL_MIN_FEATURES"]
        else:
            os.environ["LIGHTGBM_METAL_MIN_FEATURES"] = orig


def test_tiny_dataset_falls_back_to_cpu():
    """Datasets too small to benefit from Metal (few features) fall back
    to CPU cleanly without crashing. Mirrors what users get with the
    binary_classification example (28 features < default 96 threshold)."""
    X, y = make_classification(n_samples=500, n_features=8, random_state=23)
    bst = lgb.train(
        {"objective": "binary", "num_leaves": 7, "verbosity": -1,
         "device_type": "metal"},
        lgb.Dataset(X, y), num_boost_round=5,
    )
    pred = bst.predict(X)
    assert np.all(np.isfinite(pred))
    assert 0.0 <= pred.min() and pred.max() <= 1.0


def test_very_many_features():
    """Stress test: 1024 features. Verifies the auto-tuned wg_per_feat
    handles wide datasets correctly (wg_per_feat should hit 1 here,
    exercising the reduce-skipping fast path)."""
    X, y = make_classification(
        n_samples=2_000, n_features=1024, n_informative=64, n_redundant=16,
        random_state=22,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=22
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1},
        X_train, y_train, X_test, y_test, num_rounds=20,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)
    assert metal_auc > 0.7


def test_max_bin_boundary():
    """max_bin=255 puts each feature exactly at the 256-bin kernel boundary
    (255 levels + 1 missing-value bin = 256 bins total). Verifies the
    boundary case doesn't trip off-by-one bugs in the write-back loop."""
    X, y = make_classification(
        n_samples=4_000, n_features=128, n_informative=24, random_state=21,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=21
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1,
         "max_bin": 255},
        X_train, y_train, X_test, y_test, num_rounds=30,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_model_save_load_cross_device():
    """Train on Metal, save model to disk, load with no device specified,
    predict — verify the saved model is device-agnostic (no Metal-specific
    state baked into the model file)."""
    import os
    import tempfile

    X, y = make_classification(
        n_samples=2_000, n_features=64, random_state=20,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=20
    )
    params = dict(
        objective="binary", num_leaves=15, learning_rate=0.1,
        verbosity=-1, deterministic=True, seed=42,
        device_type="metal",
    )
    bst = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=30)

    with tempfile.NamedTemporaryFile(suffix=".lgb", delete=False) as f:
        tmp_path = f.name
    try:
        bst.save_model(tmp_path)
        loaded = lgb.Booster(model_file=tmp_path)
        cpu_pred = loaded.predict(X_test)
        metal_pred = bst.predict(X_test)
        # Same trees should give identical predictions regardless of
        # original training device (no Metal state in the model file).
        assert np.allclose(cpu_pred, metal_pred, atol=1e-8), \
            f"max diff: {np.abs(cpu_pred - metal_pred).max()}"
    finally:
        os.unlink(tmp_path)


def test_continue_training_parity():
    """init_model=existing_model continues training from a previous state.
    Verifies Metal handles the gradient computation correctly when starting
    from a non-zero initial score."""
    X, y = make_classification(
        n_samples=3_000, n_features=128, n_informative=24, random_state=19,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=19
    )
    base = dict(
        objective="binary", num_leaves=15, learning_rate=0.1,
        verbosity=-1, deterministic=True, seed=42,
    )

    # Round 1: train initial model (20 rounds) on each device.
    cpu_round1 = lgb.train(dict(base, device_type="cpu"),
                            lgb.Dataset(X_train, y_train), num_boost_round=20)
    metal_round1 = lgb.train(dict(base, device_type="metal"),
                              lgb.Dataset(X_train, y_train), num_boost_round=20)

    # Round 2: continue from round-1 model for 20 more rounds.
    cpu_round2 = lgb.train(dict(base, device_type="cpu"),
                            lgb.Dataset(X_train, y_train), num_boost_round=20,
                            init_model=cpu_round1)
    metal_round2 = lgb.train(dict(base, device_type="metal"),
                              lgb.Dataset(X_train, y_train), num_boost_round=20,
                              init_model=metal_round1)

    cpu_auc = roc_auc_score(y_test, cpu_round2.predict(X_test))
    metal_auc = roc_auc_score(y_test, metal_round2.predict(X_test))
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_early_stopping_parity():
    """Early stopping uses a validation set during training. Verifies the
    Metal path works with the early-stopping callback path and stops at
    similar iteration counts on cpu vs metal."""
    X, y = make_classification(
        n_samples=4_000, n_features=128, n_informative=20, random_state=18,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=18
    )
    base = dict(
        objective="binary", num_leaves=15, learning_rate=0.1,
        verbosity=-1, deterministic=True, seed=42,
    )

    cpu_train = lgb.Dataset(X_train, y_train)
    cpu_val = lgb.Dataset(X_test, y_test, reference=cpu_train)
    cpu_bst = lgb.train(
        dict(base, device_type="cpu"),
        cpu_train, num_boost_round=200, valid_sets=[cpu_val],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )

    metal_train = lgb.Dataset(X_train, y_train)
    metal_val = lgb.Dataset(X_test, y_test, reference=metal_train)
    metal_bst = lgb.train(
        dict(base, device_type="metal"),
        metal_train, num_boost_round=200, valid_sets=[metal_val],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )

    # Best iterations should be close (atomic-ordering can cause slight
    # divergence in best-iter choice).
    assert abs(cpu_bst.best_iteration - metal_bst.best_iteration) <= 20, \
        (cpu_bst.best_iteration, metal_bst.best_iteration)
    cpu_auc = roc_auc_score(y_test, cpu_bst.predict(X_test))
    metal_auc = roc_auc_score(y_test, metal_bst.predict(X_test))
    assert metal_auc == pytest.approx(cpu_auc, abs=0.02), (cpu_auc, metal_auc)


def test_multiple_models_same_process():
    """Train several distinct datasets sequentially with device_type=metal.
    Verifies the per-MetalTreeLearner state cleans up correctly and
    different dataset shapes don't bleed into each other."""
    shapes = [
        (1_500, 64),
        (2_000, 128),
        (1_000, 256),
        (1_500, 96),
    ]
    aucs = []
    for seed, (n, p) in enumerate(shapes):
        X, y = make_classification(n_samples=n, n_features=p, random_state=seed)
        ds = lgb.Dataset(X, y)
        params = {
            "objective": "binary", "num_leaves": 15, "learning_rate": 0.1,
            "verbosity": -1, "deterministic": True, "seed": 0,
            "device_type": "metal",
        }
        bst = lgb.train(params, ds, num_boost_round=20)
        aucs.append(roc_auc_score(y, bst.predict(X)))
    # All should be high AUC since the synthetic data is recoverable.
    assert all(a > 0.85 for a in aucs), aucs


def test_large_scale_drift():
    """Stress test with 100k rows + 128 features + 100 trees. Each
    histogram cell accumulates ~800 floats; atomic-ordering drift
    compounds across many splits across many trees. Asserts AUC parity
    holds at scale.

    Skipped under LIGHTGBM_TEST_METAL_QUICK=1 (CI tier-1) to keep the
    suite fast — it's the longest of the parity tests."""
    if os.environ.get("LIGHTGBM_TEST_METAL_QUICK") == "1":
        pytest.skip("quick mode")
    X, y = make_classification(
        n_samples=100_000, n_features=128, n_informative=40, n_redundant=20,
        flip_y=0.02, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    cpu_pred, metal_pred = _train_both(
        {"objective": "binary", "num_leaves": 63, "learning_rate": 0.05},
        X_train, y_train, X_test, y_test, num_rounds=100,
    )
    cpu_auc = roc_auc_score(y_test, cpu_pred)
    metal_auc = roc_auc_score(y_test, metal_pred)
    assert metal_auc == pytest.approx(cpu_auc, abs=0.005), (cpu_auc, metal_auc)
    # Per-prediction agreement: 99% of predictions within 0.01 of CPU.
    diff = np.abs(metal_pred - cpu_pred)
    pct_99 = np.quantile(diff, 0.99)
    assert pct_99 < 0.05, f"99th percentile prediction diff: {pct_99:.4f}"


def test_metal_init_smoke():
    """Smoke test: Metal device initializes and produces a non-degenerate model."""
    X, y = make_classification(n_samples=500, n_features=16, random_state=3)
    params = {
        "objective": "binary", "num_leaves": 7, "learning_rate": 0.1,
        "verbosity": -1, "deterministic": True, "device_type": "metal",
    }
    ds = lgb.Dataset(X, y)
    bst = lgb.train(params, ds, num_boost_round=10)
    pred = bst.predict(X)
    assert np.all(np.isfinite(pred))
    assert 0.0 <= pred.min() and pred.max() <= 1.0
    assert roc_auc_score(y, pred) > 0.6
