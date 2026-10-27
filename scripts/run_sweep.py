"""Launch a multi-seed sweep from a config in configs/, then summarize it.

A config has a `base` (TrainConfig fields shared by every run), a `grid`
(TrainConfig fields to sweep, crossed with each other), and `seeds`.
`seed_mode` decides how seeds become (init_seed, data_seed) pairs:
  tied   init_seed = data_seed = s for each s
  init   data_seed fixed at seeds[0], init_seed varies
  data   init_seed fixed at seeds[0], data_seed varies
  cross  every (init_seed, data_seed) pair

Each run is its own `src/train.py` process writing
results/<name>/runs/<run>.json. Runs whose JSON already exists are skipped, so
an interrupted sweep resumes where it stopped. Several runs share the GPU at
once (--jobs), since one micro-model leaves it mostly idle.

Outputs, per docs/metrics.md:
  results/<name>/runs.csv       one row per run
  results/<name>/aggregate.csv  one row per grid point: mean/std/min/max of
                                bits stored across seeds, success rate with a
                                Wilson interval, and epochs to target counting
                                censored runs as "cap"

Example:
    python scripts/run_sweep.py configs/pilot_size.json --jobs 6
"""

import argparse
import csv
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "src" / "train.py"

RUN_COLUMNS = [
    "num_params", "num_params_non_embedding", "max_bits", "success", "epochs_to_target",
    "epochs_run", "recall", "bits_stored", "bits_per_param", "fraction_of_max",
    "best_bits_stored", "fresh_bits_stored", "train_seconds",
]


def seed_pairs(seeds: list[int], mode: str) -> list[tuple[int, int]]:
    if mode == "tied":
        return [(s, s) for s in seeds]
    if mode == "init":
        return [(s, seeds[0]) for s in seeds]
    if mode == "data":
        return [(seeds[0], s) for s in seeds]
    if mode == "cross":
        return list(itertools.product(seeds, seeds))
    raise ValueError(f"unknown seed_mode {mode!r}")


def expand(config: dict) -> list[tuple[str, dict]]:
    """Every (run name, TrainConfig kwargs) in the sweep."""
    grid = config.get("grid", {})
    keys = list(grid)
    runs = []
    for point in itertools.product(*(grid[k] for k in keys)):
        for init_seed, data_seed in seed_pairs(config["seeds"], config.get("seed_mode", "tied")):
            args = {**config["base"], **dict(zip(keys, point)), "init_seed": init_seed, "data_seed": data_seed}
            name = "_".join(f"{k}{v}" for k, v in zip(keys, point)) + f"_i{init_seed}_d{data_seed}"
            runs.append((name, args))
    return runs


def run_one(name: str, args: dict, out_dir: Path) -> tuple[str, int, float]:
    out = out_dir / "runs" / f"{name}.json"
    cmd = [sys.executable, str(TRAIN), "--out", str(out)]
    for k, v in args.items():
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        else:
            cmd += [f"--{k}", str(v)]
    log = out_dir / "logs" / f"{name}.log"
    # Required for torch.use_deterministic_algorithms with cuBLAS.
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    start = time.time()
    with log.open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return name, code, time.time() - start


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(config: dict, out_dir: Path) -> list[dict]:
    keys = list(config.get("grid", {}))
    rows = []
    for name, args in expand(config):
        path = out_dir / "runs" / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        row = {k: args[k] for k in keys}
        row.update(init_seed=args["init_seed"], data_seed=args["data_seed"])
        row.update({k: result[k] for k in RUN_COLUMNS})
        rows.append(row)
    if not rows:
        return []
    with (out_dir / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for point, group in itertools.groupby(rows, key=lambda r: tuple(r[k] for k in keys)):
        group = list(group)
        bits = [r["bits_stored"] for r in group]
        successes = sum(r["success"] for r in group)
        lo, hi = wilson(successes, len(group))
        epochs = sorted((r["epochs_to_target"] for r in group), key=lambda e: math.inf if e == "cap" else e)
        aggregate.append({
            **dict(zip(keys, point)),
            "num_params": group[0]["num_params"],
            "seeds": len(group),
            "max_bits": group[0]["max_bits"],
            "bits_mean": statistics.mean(bits),
            "bits_std": statistics.stdev(bits) if len(bits) > 1 else 0.0,
            "bits_min": min(bits),
            "bits_max": max(bits),
            "bits_per_param_mean": statistics.mean(bits) / group[0]["num_params"],
            "recall_mean": statistics.mean(r["recall"] for r in group),
            "success_rate": successes / len(group),
            "success_ci_low": lo,
            "success_ci_high": hi,
            "epochs_to_target": " ".join(str(e) for e in epochs),
            "fresh_bits_max": max(r["fresh_bits_stored"] for r in group),
        })
    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    return aggregate


def print_table(aggregate: list[dict], keys: list[str]) -> None:
    header = f"{' '.join(f'{k:>8}' for k in keys)} {'params':>8} {'bits mean±std':>17} {'bits/param':>10} {'recall':>7} {'success':>8}  epochs"
    print(header)
    for a in aggregate:
        point = " ".join(f"{a[k]:>8}" for k in keys)
        print(
            f"{point} {a['num_params']:>8} {a['bits_mean']:>9.0f}±{a['bits_std']:<7.0f} "
            f"{a['bits_per_param_mean']:>10.3f} {a['recall_mean']:>7.3f} "
            f"{int(a['success_rate'] * a['seeds'])}/{a['seeds']:<6}  {a['epochs_to_target']}"
        )
    if aggregate:
        print(f"max bits per run: {aggregate[0]['max_bits']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--jobs", type=int, default=4, help="runs in parallel")
    parser.add_argument("--out", type=Path, help="default: results/<config name>")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    out_dir = args.out or ROOT / "results" / config["name"]
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    if not args.summarize_only:
        runs = expand(config)
        todo = [(n, a) for n, a in runs if not (out_dir / "runs" / f"{n}.json").exists()]
        print(f"{len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to go, {args.jobs} at a time", flush=True)
        failed = []
        with ThreadPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(run_one, n, a, out_dir) for n, a in todo]
            for i, future in enumerate(as_completed(futures), 1):
                name, code, seconds = future.result()
                status = "ok" if code == 0 else f"FAILED (exit {code}, see logs/{name}.log)"
                print(f"[{i}/{len(todo)}] {name}: {status} in {seconds:.0f}s", flush=True)
                if code != 0:
                    failed.append(name)
        if failed:
            print(f"{len(failed)} runs failed: {' '.join(failed)}")

    print_table(summarize(config, out_dir), list(config.get("grid", {})))


if __name__ == "__main__":
    main()
