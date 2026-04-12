#!/usr/bin/env python
"""Single-file inference script for unified mood classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from src.models.unified_model import UnifiedMoodModel


FEATURE_ORDER = ("mert", "mel", "chroma", "tempogram")
VALID_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
DEFAULT_MTG_THRESHOLD = 0.5


class InferenceError(RuntimeError):
    """Raised when inference setup or inputs are invalid."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _pick_default_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _default_audio_path() -> Path:
    base_dir = _repo_root()
    return _pick_default_path(
        [
            base_dir / "data" / "DEAM" / "audio" / "1769.mp3",
            base_dir / "data" / "DEAM" / "audio" / "10.mp3",
        ]
    )


def _default_checkpoint_path() -> Path:
    base_dir = _repo_root()
    return _pick_default_path(
        [
            base_dir
            / "runs"
            / "experiment_suite_20260308_130926"
            / "early_baseline"
            / "unified_mood_model_early"
            / "checkpoints"
            / "checkpoint_best.pt",
            base_dir
            / "runs"
            / "experiment_suite_20260308_230655"
            / "late_baseline"
            / "unified_mood_model_late"
            / "checkpoints"
            / "checkpoint_best.pt",
        ]
    )


def _torch_load(path: str | os.PathLike[str], **kwargs):
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def _normalize_enabled_features(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return FEATURE_ORDER

    if isinstance(raw_value, str):
        tokens = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    else:
        tokens = [str(part).strip().lower() for part in raw_value if str(part).strip()]

    if not tokens:
        raise InferenceError("Checkpoint contains an empty enabled_features configuration.")

    unknown = [token for token in tokens if token not in FEATURE_ORDER]
    if unknown:
        raise InferenceError(
            f"Checkpoint contains unknown feature(s): {unknown}. "
            f"Supported features: {', '.join(FEATURE_ORDER)}."
        )

    normalized = tuple(feature for feature in FEATURE_ORDER if feature in tokens)
    if not normalized:
        raise InferenceError("At least one feature must be enabled for inference.")
    return normalized


def _sanitize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _ensure_mood_tags(raw_tags: Any, expected_count: int) -> list[str]:
    tags = list(raw_tags or [])
    if len(tags) == expected_count:
        return tags

    if len(tags) > expected_count:
        return tags[:expected_count]

    tags.extend(f"tag_{idx}" for idx in range(len(tags), expected_count))
    return tags


def _parse_checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args") or {}
    fusion_type = checkpoint.get("fusion_type") or args.get("fusion_type") or "early"
    num_mtg_tags = int(checkpoint.get("num_mtg_tags") or len(checkpoint.get("mood_tags") or []) or 18)
    enabled_features = _normalize_enabled_features(
        checkpoint.get("enabled_features", args.get("enabled_features"))
    )

    threshold = checkpoint.get("best_mtg_threshold", DEFAULT_MTG_THRESHOLD)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = DEFAULT_MTG_THRESHOLD
    if not math.isfinite(threshold):
        threshold = DEFAULT_MTG_THRESHOLD

    return {
        "fusion_type": fusion_type,
        "hidden_dim": int(args.get("hidden_dim", 512)),
        "num_transformer_layers": int(args.get("num_transformer_layers", 4)),
        "num_heads": int(args.get("num_heads", 8)),
        "num_mtg_tags": num_mtg_tags,
        "enabled_features": enabled_features,
        "mood_tags": _ensure_mood_tags(checkpoint.get("mood_tags"), num_mtg_tags),
        "mtg_threshold": threshold,
    }


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_existing_file(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise InferenceError(f"{label} does not exist: {path}")
    return path


def _validate_audio_path(audio_path: str) -> Path:
    path = _validate_existing_file(audio_path, "Audio file")
    if path.suffix.lower() not in VALID_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(VALID_AUDIO_EXTENSIONS))
        raise InferenceError(
            f"Unsupported audio extension '{path.suffix}'. Supported extensions: {allowed}."
        )
    return path


@lru_cache(maxsize=1)
def _load_deam_index() -> dict[str, dict[str, Any]]:
    base_dir = _repo_root()
    split_path = base_dir / "data" / "DEAM" / "deam_split.json"
    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)

    labels = {}
    label_dir = base_dir / "data" / "DEAM" / "annotations" / "annotations averaged per song" / "song_level"
    for csv_name in (
        "static_annotations_averaged_songs_1_2000.csv",
        "static_annotations_averaged_songs_2000_2058.csv",
    ):
        with (label_dir / csv_name).open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                labels[row[0]] = {
                    "valence": float(row[1]),
                    "arousal": float(row[3]),
                }

    index = {}
    for split_name, audio_ids in split_data.items():
        if split_name not in {"train", "val", "test"} or not isinstance(audio_ids, list):
            continue
        for audio_id in audio_ids:
            audio_path = (base_dir / "data" / "DEAM" / "audio" / f"{audio_id}.mp3").resolve()
            index[str(audio_path)] = {
                "dataset": "deam",
                "split": split_name,
                "sample_id": str(audio_id),
                "cache_path": str(base_dir / "data" / "features" / "deam" / f"{audio_id}.pt"),
                "ground_truth": labels.get(str(audio_id)),
            }
    return index


@lru_cache(maxsize=1)
def _load_mtg_index() -> dict[str, dict[str, Any]]:
    base_dir = _repo_root()
    csv_path = base_dir / "data" / "MTG-Jamendo" / "mtg_labels.csv"
    index = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_name = (row.get("split") or "").strip()
            track_id = (row.get("track_id") or "").strip()
            audio_rel = (row.get("audio_path") or "").strip()
            if not split_name or not track_id or not audio_rel:
                continue

            audio_path = (base_dir / "data" / "MTG-Jamendo" / audio_rel).resolve()
            ground_truth = [tag for tag in (row.get("mood_tags") or "").split("|") if tag]
            index[str(audio_path)] = {
                "dataset": "mtg-jamendo",
                "split": split_name,
                "sample_id": track_id,
                "cache_path": str(base_dir / "data" / "features" / "mtg" / split_name / f"{track_id}.pt"),
                "ground_truth": ground_truth,
            }
    return index


def _resolve_audio_metadata(audio_path: Path) -> dict[str, Any]:
    resolved = str(audio_path.resolve())
    deam_match = _load_deam_index().get(resolved)
    if deam_match is not None:
        return deam_match

    mtg_match = _load_mtg_index().get(resolved)
    if mtg_match is not None:
        return mtg_match

    return {
        "dataset": "external",
        "split": None,
        "sample_id": None,
        "cache_path": None,
        "ground_truth": None,
    }


def _resolve_task(requested_task: str, metadata: dict[str, Any]) -> str:
    if requested_task != "auto":
        return requested_task

    dataset_name = metadata.get("dataset")
    if dataset_name == "deam":
        return "deam"
    if dataset_name == "mtg-jamendo":
        return "mtg"

    raise InferenceError(
        "This audio file does not match a known DEAM or MTG-Jamendo sample, so --task auto "
        "cannot infer which output to show. Please rerun with --task deam, --task mtg, or --task both."
    )


def _load_features(audio_path: Path, metadata: dict[str, Any], model_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    cache_path = metadata.get("cache_path")
    if cache_path and Path(cache_path).is_file():
        return _torch_load(cache_path, map_location="cpu", weights_only=True)

    from precompute_features import extract_features_for_audio, load_mert_model

    processor, mert_model = load_mert_model(str(model_dir), device)
    return extract_features_for_audio(str(audio_path), processor, mert_model, device)


def _prepare_single_sample(features: dict[str, torch.Tensor], enabled_features: tuple[str, ...]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    selected = {}
    missing = []
    for feature_name in enabled_features:
        tensor = features.get(feature_name)
        if tensor is None:
            missing.append(feature_name)
            continue
        selected[feature_name] = tensor

    if missing:
        raise InferenceError(
            f"Feature extraction is missing required feature(s): {', '.join(missing)}."
        )

    min_length = min(tensor.shape[0] for tensor in selected.values())
    if min_length <= 0:
        raise InferenceError("Extracted features are empty and cannot be used for inference.")

    batch = {}
    for feature_name, tensor in selected.items():
        trimmed = tensor[:min_length]
        if trimmed.dtype != torch.float32:
            trimmed = trimmed.float()
        batch[feature_name] = trimmed.unsqueeze(0)

    lengths = torch.tensor([min_length], dtype=torch.int32)
    return batch, lengths


def _predict(
    model: UnifiedMoodModel,
    features: dict[str, torch.Tensor],
    lengths: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = {key: value.to(device) for key, value in features.items()}
    batch_lengths = lengths.to(device)
    with torch.no_grad():
        outputs = model(batch, lengths=batch_lengths)
    return {key: value.detach().cpu() for key, value in outputs.items()}


def _select_mtg_tags(
    probabilities: torch.Tensor,
    mood_tags: list[str],
    threshold: float,
) -> list[tuple[str, float]]:
    probs = probabilities.tolist()
    selected = [
        (mood_tags[idx], float(prob))
        for idx, prob in enumerate(probs)
        if prob >= threshold
    ]
    if selected:
        return selected

    top_index = max(range(len(probs)), key=probs.__getitem__)
    return [(mood_tags[top_index], float(probs[top_index]))]


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[UnifiedMoodModel, dict[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise InferenceError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")

    meta = _parse_checkpoint_metadata(checkpoint)
    model = UnifiedMoodModel(
        fusion_type=meta["fusion_type"],
        hidden_dim=meta["hidden_dim"],
        num_transformer_layers=meta["num_transformer_layers"],
        num_heads=meta["num_heads"],
        num_class_tags=meta["num_mtg_tags"],
    )
    state_dict = _sanitize_state_dict_keys(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, meta


def _format_tag_entries(entries: list[tuple[str, float]]) -> str:
    return ", ".join(f"{tag} ({score:.3f})" for tag, score in entries)


def _print_results(
    audio_path: Path,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    task: str,
    outputs: dict[str, torch.Tensor],
    model_meta: dict[str, Any],
) -> None:
    print(f"Audio: {audio_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Detected dataset: {metadata['dataset']}")
    print(f"Resolved task: {task}")
    if metadata.get("split"):
        print(f"Dataset split: {metadata['split']}")
    if metadata.get("sample_id"):
        print(f"Sample ID: {metadata['sample_id']}")

    if task in {"deam", "both"}:
        regression = outputs["regression"][0]
        print(
            "Predicted VA: "
            f"valence={float(regression[0]):.4f}, arousal={float(regression[1]):.4f}"
        )
        if metadata.get("dataset") == "deam" and metadata.get("split") == "test":
            ground_truth = metadata.get("ground_truth") or {}
            print(
                "Ground Truth VA: "
                f"valence={ground_truth.get('valence', float('nan')):.4f}, "
                f"arousal={ground_truth.get('arousal', float('nan')):.4f}"
            )

    if task in {"mtg", "both"}:
        probabilities = torch.sigmoid(outputs["classification"][0])
        selected_tags = _select_mtg_tags(
            probabilities=probabilities,
            mood_tags=model_meta["mood_tags"],
            threshold=model_meta["mtg_threshold"],
        )
        print(f"MTG threshold: {model_meta['mtg_threshold']:.2f}")
        print(f"Predicted Tags: {_format_tag_entries(selected_tags)}")
        if metadata.get("dataset") == "mtg-jamendo" and metadata.get("split") == "test":
            ground_truth = metadata.get("ground_truth") or []
            print(f"Ground Truth Tags: {', '.join(ground_truth)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-audio inference for the unified mood model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio_path",
        default=str(_default_audio_path()),
        help="Path to a single audio file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_default_checkpoint_path()),
        help="Path to a trained checkpoint.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "deam", "mtg", "both"],
        help="Output mode. auto resolves from known dataset samples.",
    )
    parser.add_argument(
        "--model_dir",
        default=str(_repo_root() / "MERT"),
        help="Local MERT directory used only when cached features are unavailable.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        audio_path = _validate_audio_path(args.audio_path)
        checkpoint_path = _validate_existing_file(args.checkpoint, "Checkpoint")
        model_dir = Path(args.model_dir).expanduser().resolve()
        device = _choose_device()

        model, model_meta = _load_model(checkpoint_path, device)
        metadata = _resolve_audio_metadata(audio_path)
        task = _resolve_task(args.task, metadata)
        features = _load_features(audio_path, metadata, model_dir, device)
        batch, lengths = _prepare_single_sample(features, model_meta["enabled_features"])
        outputs = _predict(model, batch, lengths, device)
        _print_results(audio_path, checkpoint_path, metadata, task, outputs, model_meta)
    except InferenceError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
