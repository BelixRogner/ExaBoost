"""Compare CPU vs Metal training time on a meaningfully-sized dataset.

Usage:
    python tools/metal_bench/train_bench.py            # human-readable
    python tools/metal_bench/train_bench.py --csv      # CSV output
"""

import sys
import time

import numpy as np
from sklearn.datasets import make_classification

import lightgbm as lgb


CSV_MODE = "--csv" in sys.argv


def _system_info() -> str:
    import platform, subprocess
    chip = "unknown"
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    return f"{platform.system()} {platform.release()} on {chip}"


if CSV_MODE:
    print("num_data,num_features,num_iterations,device,seconds,auc")
else:
    print(f"# {_system_info()}")


def bench(num_samples: int, num_features: int, num_iterations: int = 50) -> None:
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=max(num_features // 2, 4),
        n_redundant=max(num_features // 8, 2),
        random_state=0,
    )
    if not CSV_MODE:
        print(f"\n=== {num_samples:,} rows x {num_features} features, {num_iterations} iters ===")

    base = dict(
        objective="binary",
        num_leaves=63,
        learning_rate=0.05,
        verbosity=-1,
        deterministic=True,
        seed=42,
    )

    results = {}
    for device in ("cpu", "metal"):
        params = dict(base, device_type=device)
        ds = lgb.Dataset(X, y)
        # Warmup (kernel compile + buffer alloc on Metal).
        lgb.train(dict(params, verbosity=-1), ds, num_boost_round=2)

        t0 = time.perf_counter()
        bst = lgb.train(params, ds, num_boost_round=num_iterations)
        elapsed = time.perf_counter() - t0
        pred = bst.predict(X)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, pred)
        results[device] = (elapsed, auc)
        if CSV_MODE:
            print(f"{num_samples},{num_features},{num_iterations},{device},{elapsed:.4f},{auc:.6f}")
        else:
            print(f"  {device:>5}: {elapsed:6.3f}s  AUC={auc:.6f}")
    if not CSV_MODE:
        speedup = results["cpu"][0] / results["metal"][0]
        auc_diff = abs(results["metal"][1] - results["cpu"][1])
        marker = "✓" if speedup >= 1.0 else "✗"
        print(f"  speedup: {speedup:.2f}x {marker}   AUC diff: {auc_diff:.6f}")


if __name__ == "__main__":
    bench(num_samples=500_000, num_features=64,  num_iterations=50)
    bench(num_samples=500_000, num_features=128, num_iterations=50)
    bench(num_samples=500_000, num_features=256, num_iterations=50)
    bench(num_samples=1_000_000, num_features=128, num_iterations=30)
