"""Compare CPU and CUDA training on tiny synthetic datasets.

Goal: surface every place CPU and CUDA paths diverge for matched
configurations. Tight tolerance (1e-5 raw_score) and exact tree structure.

Run:
    python scratch/cpu_cuda_parity.py
    python scratch/cpu_cuda_parity.py --only regression_basic
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

import lightgbm as lgb

RAW_TOL = 1e-5
LEAF_TOL = 1e-5
THRESH_TOL = 1e-6


def make_regression(n=200, d=8, seed=0, missing=False, sparse=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    w = rng.standard_normal(d)
    y = X @ w + 0.1 * rng.standard_normal(n)
    if missing:
        mask = rng.random(X.shape) < 0.1
        X = X.copy()
        X[mask] = np.nan
    if sparse:
        X[np.abs(X) < 0.5] = 0
        X = sp.csr_matrix(X)
    return X, y.astype(np.float64)


def make_binary(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    w = rng.standard_normal(d)
    logits = X @ w
    y = (logits > 0).astype(np.int32)
    return X, y


def make_multiclass(n=300, d=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    W = rng.standard_normal((d, k))
    logits = X @ W
    y = np.argmax(logits + 0.3 * rng.standard_normal(logits.shape), axis=1).astype(np.int32)
    return X, y


def make_categorical(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cont = rng.standard_normal((n, 4)).astype(np.float64)
    cat = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    X = np.hstack([cont, cat])
    y = (cont @ rng.standard_normal(4) + 0.5 * cat[:, 0]).astype(np.float64)
    return X, y, [4, 5]


@dataclass
class Case:
    name: str
    builder: Callable[[], tuple[Any, Any, dict]]
    params: dict
    num_round: int = 5
    note: str = ""


def _basic_data(builder):
    def fn():
        X, y = builder()
        return X, y, {}
    return fn


def _data_with_cat(builder):
    def fn():
        X, y, cats = builder()
        return X, y, {"categorical_feature": cats}
    return fn


# Common base parameters: tiny model, deterministic, double-precision GPU.
BASE = {
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_data_in_leaf": 5,
    "min_sum_hessian_in_leaf": 1e-3,
    "verbose": -1,
    "deterministic": True,
    "num_threads": 1,
    "seed": 42,
    "feature_pre_filter": False,
    "gpu_use_dp": True,
    "force_col_wise": True,
}


def case(name, builder, **overrides):
    return Case(name=name, builder=builder, params={**BASE, **overrides})


CASES: list[Case] = [
    case("reg_dense", _basic_data(lambda: make_regression(seed=1)), objective="regression"),
    case("reg_missing", _basic_data(lambda: make_regression(seed=2, missing=True)), objective="regression"),
    case("reg_sparse", _basic_data(lambda: make_regression(seed=3, sparse=True)), objective="regression"),
    case("reg_l1", _basic_data(lambda: make_regression(seed=4)), objective="regression_l1"),
    case("reg_huber", _basic_data(lambda: make_regression(seed=5)), objective="huber", alpha=1.0),
    case("reg_quantile", _basic_data(lambda: make_regression(seed=6)), objective="quantile", alpha=0.7),
    case("reg_lambda_l2", _basic_data(lambda: make_regression(seed=7)), objective="regression", lambda_l2=1.0),
    case("reg_lambda_l1", _basic_data(lambda: make_regression(seed=8)), objective="regression", lambda_l1=0.5),
    case("reg_min_leaf_1", _basic_data(lambda: make_regression(seed=9)), objective="regression", min_data_in_leaf=1),
    case("reg_feature_frac", _basic_data(lambda: make_regression(seed=10)), objective="regression", feature_fraction=0.6, feature_fraction_seed=1),
    case("reg_bagging", _basic_data(lambda: make_regression(seed=11)), objective="regression", bagging_fraction=0.7, bagging_freq=1, bagging_seed=1),
    case("reg_max_depth", _basic_data(lambda: make_regression(seed=12)), objective="regression", max_depth=3),
    case("reg_max_bin_31", _basic_data(lambda: make_regression(seed=13)), objective="regression", max_bin=31),
    case("reg_categorical", _data_with_cat(lambda: make_categorical(seed=14)), objective="regression"),
    case("bin_dense", _basic_data(lambda: make_binary(seed=20)), objective="binary"),
    case("bin_missing", _basic_data(lambda: make_binary(seed=21)), objective="binary"),
    case("bin_xentropy", _basic_data(lambda: make_binary(seed=22)), objective="cross_entropy"),
    case("multi_dense", _basic_data(lambda: make_multiclass(seed=30)), objective="multiclass", num_class=3),
    case("multi_softmax_lr", _basic_data(lambda: make_multiclass(seed=31)), objective="multiclass", num_class=3, learning_rate=0.05),
]


@dataclass
class Result:
    name: str
    ok_pred: bool
    ok_tree: bool
    max_abs_pred: float
    mean_abs_pred: float
    n_trees_cpu: int
    n_trees_cuda: int
    tree_diffs: list[str] = field(default_factory=list)
    error: str = ""


def train(params, num_round, X, y, fit_kwargs):
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False}, **fit_kwargs)
    return lgb.train(params, ds, num_boost_round=num_round)


def diff_trees(model_a, model_b) -> list[str]:
    """Compare every split + leaf in two boosters; return list of mismatch strings."""
    a = model_a.dump_model()
    b = model_b.dump_model()
    out = []
    ta = a["tree_info"]
    tb = b["tree_info"]
    if len(ta) != len(tb):
        out.append(f"tree-count: cpu={len(ta)} cuda={len(tb)}")
        return out
    for i, (ti_a, ti_b) in enumerate(zip(ta, tb)):
        out.extend(diff_node(f"tree{i}", ti_a["tree_structure"], ti_b["tree_structure"]))
    return out


def diff_node(path, na, nb):
    diffs = []
    is_leaf_a = "leaf_value" in na
    is_leaf_b = "leaf_value" in nb
    if is_leaf_a != is_leaf_b:
        diffs.append(f"{path}: structure mismatch (leaf_a={is_leaf_a} leaf_b={is_leaf_b})")
        return diffs
    if is_leaf_a:
        if abs(na["leaf_value"] - nb["leaf_value"]) > LEAF_TOL:
            diffs.append(f"{path}: leaf {na['leaf_value']:.7g} vs {nb['leaf_value']:.7g}")
        return diffs
    if na["split_feature"] != nb["split_feature"]:
        diffs.append(f"{path}: split_feature cpu={na['split_feature']} cuda={nb['split_feature']}")
        return diffs
    ta_, tb_ = na.get("threshold"), nb.get("threshold")
    if isinstance(ta_, (int, float)) and isinstance(tb_, (int, float)):
        if abs(ta_ - tb_) > THRESH_TOL:
            diffs.append(f"{path}: threshold {ta_:.7g} vs {tb_:.7g}")
    elif ta_ != tb_:
        diffs.append(f"{path}: threshold(cat) {ta_} vs {tb_}")
    if na.get("decision_type") != nb.get("decision_type"):
        diffs.append(f"{path}: decision_type {na.get('decision_type')} vs {nb.get('decision_type')}")
    if na.get("default_left") != nb.get("default_left"):
        diffs.append(f"{path}: default_left {na.get('default_left')} vs {nb.get('default_left')}")
    diffs.extend(diff_node(path + "/L", na["left_child"], nb["left_child"]))
    diffs.extend(diff_node(path + "/R", na["right_child"], nb["right_child"]))
    return diffs


def run_case(c: Case) -> Result:
    X, y, fit_kwargs = c.builder()
    try:
        cpu_params = {**c.params, "device_type": "cpu"}
        cuda_params = {**c.params, "device_type": "cuda"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cpu_model = train(cpu_params, c.num_round, X, y, fit_kwargs)
            cuda_model = train(cuda_params, c.num_round, X, y, fit_kwargs)
        # raw_score for objective-agnostic comparison
        pred_cpu = cpu_model.predict(X, raw_score=True)
        pred_cuda = cuda_model.predict(X, raw_score=True)
        if pred_cpu.shape != pred_cuda.shape:
            return Result(
                name=c.name,
                ok_pred=False,
                ok_tree=False,
                max_abs_pred=float("nan"),
                mean_abs_pred=float("nan"),
                n_trees_cpu=cpu_model.num_trees(),
                n_trees_cuda=cuda_model.num_trees(),
                error=f"shape mismatch cpu={pred_cpu.shape} cuda={pred_cuda.shape}",
            )
        diff = np.abs(pred_cpu - pred_cuda)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        tree_diffs = diff_trees(cpu_model, cuda_model)
        return Result(
            name=c.name,
            ok_pred=max_abs <= RAW_TOL,
            ok_tree=not tree_diffs,
            max_abs_pred=max_abs,
            mean_abs_pred=mean_abs,
            n_trees_cpu=cpu_model.num_trees(),
            n_trees_cuda=cuda_model.num_trees(),
            tree_diffs=tree_diffs,
        )
    except Exception as e:  # noqa: BLE001
        return Result(
            name=c.name,
            ok_pred=False,
            ok_tree=False,
            max_abs_pred=float("nan"),
            mean_abs_pred=float("nan"),
            n_trees_cpu=0,
            n_trees_cuda=0,
            error=f"{type(e).__name__}: {e}",
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run only the case with this name")
    ap.add_argument("--max-tree-diffs", type=int, default=8)
    ap.add_argument("--json", help="dump full results to json file")
    args = ap.parse_args(argv)

    cases = CASES if not args.only else [c for c in CASES if c.name == args.only]
    if not cases:
        print(f"no case named {args.only!r}; known: {[c.name for c in CASES]}")
        return 2

    results = []
    print(f"{'case':<22} {'pred':>4} {'tree':>4} {'max|Δ|':>11} {'mean|Δ|':>11} {'#cpu':>5} {'#cu':>4} note")
    print("-" * 88)
    for c in cases:
        r = run_case(c)
        results.append(r)
        if r.error:
            print(f"{r.name:<22} {'ERR':>4} {'ERR':>4} {'-':>11} {'-':>11} {'-':>5} {'-':>4} {r.error}")
            continue
        pred_mark = "OK" if r.ok_pred else "X"
        tree_mark = "OK" if r.ok_tree else "X"
        print(
            f"{r.name:<22} {pred_mark:>4} {tree_mark:>4} "
            f"{r.max_abs_pred:>11.3e} {r.mean_abs_pred:>11.3e} "
            f"{r.n_trees_cpu:>5} {r.n_trees_cuda:>4} "
            f"{'tree-diffs=' + str(len(r.tree_diffs)) if r.tree_diffs else ''}"
        )
    print()
    any_diffs = False
    for r in results:
        if r.tree_diffs:
            any_diffs = True
            print(f"== {r.name}: first {min(args.max_tree_diffs, len(r.tree_diffs))} of {len(r.tree_diffs)} tree differences ==")
            for d in r.tree_diffs[: args.max_tree_diffs]:
                print(f"  {d}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
    bad = [r for r in results if not (r.ok_pred and r.ok_tree) or r.error]
    print(f"\nsummary: {len(results) - len(bad)}/{len(results)} clean, {len(bad)} divergent or errored")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
