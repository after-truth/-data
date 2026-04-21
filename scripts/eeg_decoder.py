from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8-sig") as f:
        current = yaml.safe_load(f) or {}

    inherits = current.pop("inherits", None)
    if not inherits:
        return current

    inherited_path = Path(inherits)
    if inherited_path.is_absolute():
        parent_path = inherited_path
    else:
        cand1 = (config_path.parent / inherited_path).resolve()
        cand2 = (Path.cwd() / inherited_path).resolve()
        parent_path = cand1 if cand1.exists() else cand2

    if not parent_path.exists():
        raise FileNotFoundError(f"Inherited config not found: {inherits}")

    parent = load_config(parent_path)
    return deep_merge(parent, current)


def load_triplets(root_dir: Path, target_type: str, exclude_keywords: List[str]) -> List[Dict]:
    records = []
    for eeg_file in sorted(root_dir.glob("*_eeg.npy")):
        if any(kw in eeg_file.name for kw in exclude_keywords):
            continue

        parts = eeg_file.stem.split("_-_")
        if len(parts) != 4 or parts[-1] != "eeg":
            continue

        prefix, subject, recording, _ = parts
        target_file = eeg_file.with_name(f"{prefix}_-_{subject}_-_{recording}_-_{target_type}.npy")
        if not target_file.exists():
            continue

        eeg = np.load(eeg_file).astype(np.float32)
        target = np.load(target_file).astype(np.float32)

        n = min(len(eeg), len(target))
        if n <= 0:
            continue

        records.append(
            {
                "subject": subject,
                "recording": recording,
                "eeg": eeg[:n],
                "target": target[:n],
            }
        )
    return records


def split_records(records: List[Dict], holdout_subject: str) -> Tuple[List[Dict], List[Dict]]:
    train = [r for r in records if r["subject"] != holdout_subject]
    test = [r for r in records if r["subject"] == holdout_subject]
    return train, test


def window_recording(
    eeg: np.ndarray,
    target: np.ndarray,
    window_size: int,
    hop_size: int,
    max_windows: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    end_limit = len(eeg) - window_size + 1
    for s in range(0, max(0, end_limit), hop_size):
        e = s + window_size
        xs.append(eeg[s:e])
        ys.append(target[s:e].mean(axis=0))

    if not xs:
        return (
            np.empty((0, window_size, eeg.shape[1]), dtype=np.float32),
            np.empty((0, target.shape[1]), dtype=np.float32),
        )

    x = np.stack(xs)
    y = np.stack(ys)

    if max_windows > 0 and len(x) > max_windows:
        idx = np.sort(rng.choice(len(x), size=max_windows, replace=False))
        x = x[idx]
        y = y[idx]
    return x, y


def extract_features(windows: np.ndarray, feature_mode: str, local_context: int) -> np.ndarray:
    if windows.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    mean_feat = windows.mean(axis=1)
    state = np.zeros_like(mean_feat)
    alpha = 0.9
    for t in range(windows.shape[1]):
        state = alpha * state + (1.0 - alpha) * windows[:, t, :]

    if feature_mode == "ssm_only":
        return np.concatenate([state, mean_feat], axis=1)

    ctx = min(local_context, windows.shape[1])
    context_feat = windows[:, -ctx:, :].mean(axis=1)
    std_feat = windows.std(axis=1)
    return np.concatenate([mean_feat, std_feat, context_feat, state], axis=1)


def build_dataset(records: List[Dict], config: Dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    task = config["task"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    rng = np.random.default_rng(seed)
    all_x, all_y = [], []

    for r in records:
        w, y = window_recording(
            eeg=r["eeg"],
            target=r["target"],
            window_size=int(task.get("window_size", 256)),
            hop_size=int(task.get("hop_size", 64)),
            max_windows=int(train_cfg.get("max_windows_per_recording", 0)),
            rng=rng,
        )
        if len(w) == 0:
            continue
        x = extract_features(
            windows=w,
            feature_mode=model_cfg.get("feature_mode", "fusion"),
            local_context=int(model_cfg.get("local_context", 32)),
        )
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class LinearDecoder:
    def __init__(self, weight_decay: float) -> None:
        self.weight_decay = float(weight_decay)
        self.w = None
        self.b = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
        reg = np.eye(xb.shape[1], dtype=x.dtype) * self.weight_decay
        reg[-1, -1] = 0.0
        theta = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"w": self.w, "b": self.b}


class MLPDecoder:
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, lr: float, wd: float, epochs: int, batch: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = (rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros((out_dim,), dtype=np.float32)
        self.lr = float(lr)
        self.wd = float(wd)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.seed = int(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n = len(x)
        for _ in range(self.epochs):
            idx_all = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = idx_all[s : s + self.batch]
                xb, yb = x[idx], y[idx]

                z1 = xb @ self.w1 + self.b1
                h1 = np.maximum(z1, 0.0)
                pred = h1 @ self.w2 + self.b2

                g_out = (pred - yb) * (2.0 / len(xb))
                g_w2 = h1.T @ g_out + self.wd * self.w2
                g_b2 = g_out.sum(axis=0)

                g_h1 = g_out @ self.w2.T
                g_z1 = g_h1 * (z1 > 0.0)
                g_w1 = xb.T @ g_z1 + self.wd * self.w1
                g_b1 = g_z1.sum(axis=0)

                self.w2 -= self.lr * g_w2
                self.b2 -= self.lr * g_b2
                self.w1 -= self.lr * g_w1
                self.b1 -= self.lr * g_b1

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return h @ self.w2 + self.b2

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}


def build_decoder(config: Dict, in_dim: int, out_dim: int):
    model_cfg = config["model"]
    train_cfg = config["training"]
    typ = model_cfg.get("decoder_type", "linear")
    if typ == "mlp":
        return MLPDecoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=int(model_cfg.get("mlp_hidden_dim", 128)),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            wd=float(train_cfg.get("weight_decay", 1e-4)),
            epochs=int(train_cfg.get("epochs", 20)),
            batch=int(train_cfg.get("batch_size", 16)),
            seed=int(config.get("seed", 42)),
        )
    return LinearDecoder(weight_decay=float(train_cfg.get("weight_decay", 1e-4)))


def apply_subject_bias(pred: np.ndarray, target: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    ratio = max(0.0, min(0.9, float(ratio)))
    n = max(1, int(len(target) * ratio))
    bias = target[:n].mean(axis=0) - pred[:n].mean(axis=0)
    return pred[n:] + bias, target[n:]


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))

    cors = []
    for i in range(target.shape[1]):
        x = pred[:, i]
        y = target[:, i]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(x, y)[0, 1]))

    return {"mse": mse, "mae": mae, "corr": float(np.mean(cors))}


def save_artifacts(run_dir: Path, config: Dict, scaler: StandardScaler, decoder, metrics: Dict[str, float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=False, sort_keys=False)

    state = decoder.state_dict()
    state["x_mean"] = scaler.mean
    state["x_std"] = scaler.std
    np.savez(run_dir / "model.npz", **state)

    meta = {
        "decoder_type": config["model"].get("decoder_type", "linear"),
        "feature_mode": config["model"].get("feature_mode", "fusion"),
        "target_type": config["task"].get("target_type", "mel"),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
