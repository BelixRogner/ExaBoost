"""Benchmark ExaBoost CPU vs Metal on Numerai v5.2 training data.

Requires /tmp/numerai_data/{train.parquet, features.json} (downloaded via
the numerapi CLI). Trains identical regression models on each device and
reports wall-clock time and per-era correlation (the Numerai metric).

Usage:
    python tools/metal_bench/numerai_bench.py            [medium feature set, ~500k rows subset]
    python tools/metal_bench/numerai_bench.py --all-rows [full 2.75M rows]
    python tools/metal_bench/numerai_bench.py --all      [all 2748 features]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr

import lightgbm as lgb


DATA_DIR = Path("/tmp/numerai_data")


def load_data(feature_set: str, max_rows: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, eras). Features as int8; target as float32."""
    with open(DATA_DIR / "features.json") as f:
        meta = json.load(f)
    features = meta["feature_sets"][feature_set]
    target = "target_cyrusd_20"  # v5.2 default-ish target
    print(f"feature_set={feature_set!r}: {len(features)} features")

    t0 = time.perf_counter()
    cols = ["era", target] + features
    table = pq.read_table(DATA_DIR / "train.parquet", columns=cols)
    if max_rows is not None and table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    df = table.to_pandas()
    print(f"loaded {len(df):,} rows in {time.perf_counter() - t0:.1f}s")
    # Drop rows missing target
    df = df.dropna(subset=[target])
    print(f"after dropna: {len(df):,} rows")

    X = df[features].to_numpy(dtype=np.int8)  # Numerai bins as 0..4
    y = df[target].to_numpy(dtype=np.float32)
    eras = df["era"].to_numpy()
    return X, y, eras


def per_era_correlation(y_true: np.ndarray, y_pred: np.ndarray, eras: np.ndarray) -> float:
    """Mean per-era Pearson correlation (Numerai's headline metric)."""
    corrs = []
    for era in np.unique(eras):
        m = eras == era
        if m.sum() < 2:
            continue
        if np.std(y_pred[m]) == 0 or np.std(y_true[m]) == 0:
            continue
        corrs.append(pearsonr(y_pred[m], y_true[m]).statistic)
    return float(np.mean(corrs))


def run_bench(feature_set: str, max_rows: int | None, num_iter: int) -> None:
    X, y, eras = load_data(feature_set, max_rows)
    print(f"X.shape={X.shape}  y.shape={y.shape}")
    print(f"unique eras: {len(np.unique(eras))}")

    base_params = dict(
        objective="regression",
        num_leaves=63,
        learning_rate=0.05,
        verbosity=-1,
        deterministic=True,
        seed=42,
        feature_fraction=1.0,
    )

    results = {}
    for device in ("cpu", "metal"):
        params = dict(base_params, device_type=device)
        ds = lgb.Dataset(X, y)
        # Warmup (1 round for Metal kernel/PSO compile if needed).
        lgb.train(dict(params, verbosity=-1), ds, num_boost_round=1)

        ds = lgb.Dataset(X, y)
        t0 = time.perf_counter()
        bst = lgb.train(params, ds, num_boost_round=num_iter)
        elapsed = time.perf_counter() - t0

        # Predict + evaluate on training (per-era correlation).
        pred = bst.predict(X)
        corr = per_era_correlation(y, pred, eras)
        results[device] = (elapsed, corr)
        print(f"  {device:>5}: {elapsed:6.1f}s  per-era-corr={corr:.5f}")

    cpu_t, metal_t = results["cpu"][0], results["metal"][0]
    cpu_c, metal_c = results["cpu"][1], results["metal"][1]
    speedup = cpu_t / metal_t if metal_t > 0 else float("nan")
    marker = "✓" if speedup >= 1.0 else "✗"
    print(f"\nMetal speedup: {speedup:.2f}x {marker}")
    print(f"Correlation diff: cpu={cpu_c:.5f}  metal={metal_c:.5f}  |Δ|={abs(metal_c - cpu_c):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="use all 2748 features (default: medium=780)")
    ap.add_argument("--all-rows", action="store_true", help="use all 2.75M rows (default: 500k subset)")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    import platform, subprocess
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    print(f"# {platform.system()} {platform.release()} on {chip}")

    run_bench(
        feature_set="all" if args.all else "medium",
        max_rows=None if args.all_rows else 500_000,
        num_iter=args.iters,
    )
