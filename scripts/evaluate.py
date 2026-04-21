from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_decoder import (
    StandardScaler,
    apply_subject_bias,
    build_dataset,
    compute_metrics,
    load_config,
    load_triplets,
    split_records,
    LinearDecoder,
    MLPDecoder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained EEG decoder")
    p.add_argument("--config", required=True, help="Path to yaml config")
    p.add_argument("--run-dir", default="", help="Optional explicit run directory")
    return p.parse_args()


def load_decoder_from_npz(run_dir: Path, decoder_type: str):
    state = np.load(run_dir / "model.npz")

    if decoder_type == "mlp":
        decoder = MLPDecoder(
            in_dim=int(state["w1"].shape[0]),
            out_dim=int(state["w2"].shape[1]),
            hidden_dim=int(state["w1"].shape[1]),
            lr=1e-3,
            wd=0.0,
            epochs=1,
            batch=16,
            seed=42,
        )
        decoder.w1 = state["w1"]
        decoder.b1 = state["b1"]
        decoder.w2 = state["w2"]
        decoder.b2 = state["b2"]
    else:
        decoder = LinearDecoder(weight_decay=0.0)
        decoder.w = state["w"]
        decoder.b = state["b"]

    scaler = StandardScaler(mean=state["x_mean"], std=state["x_std"])
    return decoder, scaler


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        log_cfg = config.get("logging", {})
        run_name = log_cfg.get("run_name", "baseline")
        save_dir = Path(log_cfg.get("save_dir", "results"))
        run_dir = (save_dir / run_name).resolve()

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found in {run_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    decoder_type = meta.get("decoder_type", "linear")
    decoder, scaler = load_decoder_from_npz(run_dir, decoder_type)

    data_cfg = config["data"]
    root_dir = Path(data_cfg.get("root_dir", ".")).resolve()
    target_type = config["task"].get("target_type", "mel")
    exclude_keywords = list(data_cfg.get("exclude_keywords", []))

    records = load_triplets(root_dir, target_type, exclude_keywords)

    holdout = config.get("protocol", {}).get("holdout_subject")
    if not holdout:
        raise RuntimeError("protocol.holdout_subject is required for evaluation.")

    _, test_records = split_records(records, holdout)
    x_test, y_test = build_dataset(test_records, config, int(config.get("seed", 42)) + 1)
    if len(x_test) == 0:
        raise RuntimeError("No test windows generated.")

    x_test_s = scaler.transform(x_test)
    y_pred = decoder.predict(x_test_s)

    p_cfg = config.get("personalization", {})
    if p_cfg.get("enabled", False) and p_cfg.get("strategy") == "subject_bias":
        ratio = float(p_cfg.get("fine_tune_ratio", 0.1))
        y_pred, y_test = apply_subject_bias(y_pred, y_test, ratio)

    metrics = compute_metrics(y_pred, y_test)
    metrics["num_test_windows"] = int(len(x_test))

    with (run_dir / "metrics_eval.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Evaluated run: {run_dir}")
    print(f"MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} CORR={metrics['corr']:.6f}")


if __name__ == "__main__":
    main()
