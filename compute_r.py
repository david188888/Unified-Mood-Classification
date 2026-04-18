#!/usr/bin/env python
"""Offline checkpoint-based DEAM R^2 recomputation and sanity-check utility."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataloader_fast import get_dataloader_fast
from src.models.unified_model import UnifiedMoodModel


FEATURE_ORDER = ("mert", "mel", "chroma", "tempogram")


def _default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_enabled_features(raw_value) -> tuple[str, ...]:
    if raw_value is None:
        return FEATURE_ORDER

    if isinstance(raw_value, str):
        tokens = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    else:
        tokens = [str(part).strip().lower() for part in raw_value if str(part).strip()]

    unknown = [token for token in tokens if token not in FEATURE_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown feature(s): {unknown}. Allowed values: {', '.join(FEATURE_ORDER)}"
        )

    normalized = tuple(feature for feature in FEATURE_ORDER if feature in tokens)
    if not normalized:
        raise ValueError("At least one feature must be enabled")
    return normalized


def _filter_enabled_features(
    features: dict[str, torch.Tensor],
    enabled_features: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {key: value for key, value in features.items() if key in enabled_features}


def _r2_from_predictions(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape:
        raise ValueError(
            f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}"
        )

    residual_sum = np.sum((target - prediction) ** 2, dtype=np.float64)
    total_sum = np.sum((target - np.mean(target, dtype=np.float64)) ** 2, dtype=np.float64)
    if total_sum <= 0.0:
        return float("nan")
    return float(1.0 - residual_sum / total_sum)


def compute_r2_from_tensors(pred_path: str, label_path: str) -> tuple[float, float, float]:
    preds = torch.load(pred_path, map_location="cpu", weights_only=False)
    labels = torch.load(label_path, map_location="cpu", weights_only=False)

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    else:
        preds = np.asarray(preds)

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    else:
        labels = np.asarray(labels)

    r2_v = _r2_from_predictions(labels[:, 0], preds[:, 0])
    r2_a = _r2_from_predictions(labels[:, 1], preds[:, 1])
    return r2_v, r2_a, float(np.nanmean([r2_v, r2_a]))


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> tuple[UnifiedMoodModel, tuple[str, ...], dict]:
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        raise TypeError(f"checkpoint['args'] must be a dict, got {type(args)!r}")

    fusion_type = checkpoint.get("fusion_type", args.get("fusion_type", "early"))
    hidden_dim = int(args.get("hidden_dim", 512))
    num_transformer_layers = int(args.get("num_transformer_layers", 4))
    num_heads = int(args.get("num_heads", 8))
    num_mtg_tags = int(checkpoint.get("num_mtg_tags", len(checkpoint.get("mood_tags", [])) or 18))
    enabled_features = _normalize_enabled_features(
        checkpoint.get("enabled_features", args.get("enabled_features"))
    )

    model = UnifiedMoodModel(
        fusion_type=fusion_type,
        hidden_dim=hidden_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        num_class_tags=num_mtg_tags,
        deam_v_range=(1.6, 8.4),
        deam_a_range=(1.6, 8.2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = {
        "fusion_type": fusion_type,
        "hidden_dim": hidden_dim,
        "num_transformer_layers": num_transformer_layers,
        "num_heads": num_heads,
        "num_mtg_tags": num_mtg_tags,
    }
    return model, enabled_features, config


def compute_r2_from_checkpoint(
    checkpoint_path: str,
    split: str = "test",
    batch_size: int | None = None,
    num_workers: int = 0,
    device: str | None = None,
    save_dir: str | None = None,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(device) if device else _default_device()
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)
    model, enabled_features, config = _build_model_from_checkpoint(checkpoint, resolved_device)

    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}

    loader_batch_size = int(batch_size or args.get("batch_size", 8))
    loader = get_dataloader_fast(
        "deam",
        split=split,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        enabled_features=enabled_features,
    )

    all_preds = []
    all_labels = []

    use_amp = resolved_device.type == "mps"
    autocast_ctx = (
        torch.amp.autocast(device_type="mps", dtype=torch.float16)
        if use_amp
        else torch.amp.autocast(device_type=resolved_device.type, enabled=False)
    )

    with torch.no_grad():
        for features, labels, feat_lengths in loader:
            for key in features:
                features[key] = features[key].to(resolved_device)
            labels = labels.to(resolved_device)
            feat_lengths = feat_lengths.to(resolved_device)
            features = _filter_enabled_features(features, enabled_features)

            with autocast_ctx:
                outputs = model(features, lengths=feat_lengths)

            all_preds.append(outputs["regression"].detach().cpu())
            all_labels.append(labels.detach().cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    r2_valence = _r2_from_predictions(labels[:, 0], preds[:, 0])
    r2_arousal = _r2_from_predictions(labels[:, 1], preds[:, 1])
    r2_mean = float(np.nanmean([r2_valence, r2_arousal]))

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = checkpoint_path.parent.parent.parent.name
        torch.save(torch.from_numpy(preds), output_dir / f"{stem}_{split}_preds.pt")
        torch.save(torch.from_numpy(labels), output_dir / f"{stem}_{split}_labels.pt")

    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "device": str(resolved_device),
        "batch_size": loader_batch_size,
        "num_samples": int(labels.shape[0]),
        "enabled_features": enabled_features,
        "config": config,
        "r2_valence": r2_valence,
        "r2_arousal": r2_arousal,
        "r2_mean": r2_mean,
    }


def _format_result(result: dict[str, object]) -> str:
    config = result["config"]
    enabled = ",".join(result["enabled_features"])
    return (
        f"checkpoint={result['checkpoint']}\n"
        f"split={result['split']} device={result['device']} samples={result['num_samples']} "
        f"batch_size={result['batch_size']}\n"
        f"fusion={config['fusion_type']} hidden_dim={config['hidden_dim']} "
        f"layers={config['num_transformer_layers']} heads={config['num_heads']} "
        f"enabled_features={enabled}\n"
        f"R^2 Valence={result['r2_valence']:.6f}\n"
        f"R^2 Arousal={result['r2_arousal']:.6f}\n"
        f"R^2 Mean={result['r2_mean']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DEAM R^2 from a checkpoint or saved tensors")
    parser.add_argument(
        "checkpoints",
        nargs="*",
        help="One or more checkpoint paths. If omitted, uses the baseline early/late best checkpoints.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None, help="Optional directory to save preds/labels tensors")
    parser.add_argument("--pred_path", type=str, default=None, help="Optional saved prediction tensor path")
    parser.add_argument("--label_path", type=str, default=None, help="Optional saved label tensor path")

    args = parser.parse_args()

    if args.pred_path or args.label_path:
        if not (args.pred_path and args.label_path):
            raise ValueError("--pred_path and --label_path must be provided together")
        r2_v, r2_a, r2_mean = compute_r2_from_tensors(args.pred_path, args.label_path)
        print(f"R^2 Valence={r2_v:.6f}")
        print(f"R^2 Arousal={r2_a:.6f}")
        print(f"R^2 Mean={r2_mean:.6f}")
        return

    checkpoints = args.checkpoints or [
        "runs/experiment_suite_20260308_130926/early_baseline/unified_mood_model_early/checkpoints/checkpoint_best.pt",
        "runs/experiment_suite_20260308_230655/late_baseline/unified_mood_model_late/checkpoints/checkpoint_best.pt",
    ]

    for index, checkpoint in enumerate(checkpoints):
        result = compute_r2_from_checkpoint(
            checkpoint_path=checkpoint,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            save_dir=args.save_dir,
        )
        if index:
            print()
        print(_format_result(result))


if __name__ == "__main__":
    main()
