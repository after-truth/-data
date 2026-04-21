from __future__ import annotations

import argparse
from pathlib import Path

from eeg_decoder import (
    StandardScaler,
    apply_subject_bias,
    build_dataset,
    build_decoder,
    compute_metrics,
    load_config,
    load_triplets,
    save_artifacts,
    split_records,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG-to-audio decoder")
    p.add_argument("--config", required=True, help="Path to yaml config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    np_random = __import__("numpy").random
    np_random.seed(seed)

    data_cfg = config["data"]
    root_dir = Path(data_cfg.get("root_dir", ".")).resolve()
    target_type = config["task"].get("target_type", "mel")
    exclude_keywords = list(data_cfg.get("exclude_keywords", []))

    records = load_triplets(root_dir, target_type, exclude_keywords)
    if not records:
        raise RuntimeError("No valid EEG/target recordings found in root_dir.")

    holdout = config.get("protocol", {}).get("holdout_subject")
    if not holdout:
        raise RuntimeError("protocol.holdout_subject is required.")

    train_records, test_records = split_records(records, holdout)
    if not train_records or not test_records:
        raise RuntimeError("Train/test split failed. Check holdout subject and data files.")

    x_train, y_train = build_dataset(train_records, config, seed)
    x_test, y_test = build_dataset(test_records, config, seed + 1)
    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError("Dataset is empty after windowing. Check window_size/hop_size.")

    scaler = StandardScaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    decoder = build_decoder(config, in_dim=x_train_s.shape[1], out_dim=y_train.shape[1])
    decoder.fit(x_train_s, y_train)
    y_pred = decoder.predict(x_test_s)

    p_cfg = config.get("personalization", {})
    if p_cfg.get("enabled", False) and p_cfg.get("strategy") == "subject_bias":
        ratio = float(p_cfg.get("fine_tune_ratio", 0.1))
        y_pred, y_test = apply_subject_bias(y_pred, y_test, ratio)

    metrics = compute_metrics(y_pred, y_test)
    metrics["num_train_windows"] = int(len(x_train))
    metrics["num_test_windows"] = int(len(x_test))

    log_cfg = config.get("logging", {})
    run_name = log_cfg.get("run_name", "baseline")
    save_dir = Path(log_cfg.get("save_dir", "results"))
    run_dir = (save_dir / run_name).resolve()

    save_artifacts(run_dir, config, scaler, decoder, metrics)

    print(f"Saved run to: {run_dir}")
    print(f"MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} CORR={metrics['corr']:.6f}")


if __name__ == "__main__":
    main()
