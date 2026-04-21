from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize experiment metrics")
    p.add_argument("--results-dir", default="results", help="Directory containing run outputs")
    return p.parse_args()


def load_metrics(run_dir: Path):
    eval_path = run_dir / "metrics_eval.json"
    train_path = run_dir / "metrics.json"

    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return None


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise RuntimeError(f"Results directory not found: {results_dir}")

    run_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    rows = []

    for run_dir in run_dirs:
        m = load_metrics(run_dir)
        if m is None:
            continue
        rows.append((run_dir.name, m))

    if not rows:
        print("No metrics found.")
        return

    print("run_name,mse,mae,corr,num_test_windows")
    for name, m in rows:
        print(
            f"{name},"
            f"{m.get('mse', float('nan')):.6f},"
            f"{m.get('mae', float('nan')):.6f},"
            f"{m.get('corr', float('nan')):.6f},"
            f"{int(m.get('num_test_windows', 0))}"
        )


if __name__ == "__main__":
    main()
