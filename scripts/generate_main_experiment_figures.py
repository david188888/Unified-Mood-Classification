#!/usr/bin/env python3
"""Generate raw-data figures for the Early vs Late main experiments."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = [
    ("Early Fusion", "runs/experiment_suite_20260308_130926/early_baseline/unified_mood_model_early"),
    ("Late Fusion", "runs/experiment_suite_20260308_230655/late_baseline/unified_mood_model_late"),
]

COLORS = {
    "Early Fusion": "#2878B5",
    "Late Fusion": "#9AC9DB",
}

TEST_PATTERNS = {
    "test_loss": r"Test Loss: ([0-9.]+)",
    "test_deam_rmse": r"Test DEAM RMSE: ([0-9.]+)",
    "test_deam_pearson": r"Test DEAM Pearson: ([0-9.]+)",
    "test_deam_ccc_epoch": r"Test DEAM CCC \(epoch\): ([0-9.]+)",
    "test_deam_valence": r"Test DEAM CCC Valence: ([0-9.]+)",
    "test_deam_arousal": r"Test DEAM CCC Arousal: ([0-9.]+)",
    "test_mtg": (
        r"Test MTG thr=([0-9.]+) \(top1_fallback=on\) \| "
        r"ROC-AUC micro/macro: ([0-9.]+)/([0-9.]+) \| "
        r"PR-AUC micro/macro: ([0-9.]+)/([0-9.]+) \| "
        r"F1 micro/macro: ([0-9.]+)/([0-9.]+) \| "
        r"P micro/macro: ([0-9.]+)/([0-9.]+) \| "
        r"R micro/macro: ([0-9.]+)/([0-9.]+)"
    ),
}


def _read_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    data: dict[str, list[float]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            if key == "epoch":
                data[key].append(int(float(value)))
            else:
                data[key].append(float(value))
    return {key: np.asarray(values) for key, values in data.items()}


def _parse_test_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text()
    parsed: dict[str, float] = {}

    for key, pattern in TEST_PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            continue
        if key == "test_mtg":
            values = [float(item) for item in match.groups()]
            parsed.update(
                {
                    "test_mtg_threshold": values[0],
                    "test_mtg_roc_auc_micro": values[1],
                    "test_mtg_roc_auc_macro": values[2],
                    "test_mtg_pr_auc_micro": values[3],
                    "test_mtg_pr_auc_macro": values[4],
                    "test_mtg_f1_micro": values[5],
                    "test_mtg_f1_macro": values[6],
                    "test_mtg_precision_micro": values[7],
                    "test_mtg_precision_macro": values[8],
                    "test_mtg_recall_micro": values[9],
                    "test_mtg_recall_macro": values[10],
                }
            )
        else:
            parsed[key] = float(match.group(1))

    return parsed


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dpi = 300 if ext == "png" else None
        fig.savefig(out_path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _load_experiments() -> tuple[list[dict[str, float | str]], dict[str, dict[str, np.ndarray]]]:
    records: list[dict[str, float | str]] = []
    histories: dict[str, dict[str, np.ndarray]] = {}

    for label, run_dir_raw in EXPERIMENTS:
        run_dir = Path(run_dir_raw)
        metrics = _read_metrics(run_dir / "metrics.csv")
        test_metrics = _parse_test_metrics(run_dir / "training.log")

        best_val_idx = int(np.argmin(metrics["val_loss"]))
        best_deam_idx = int(np.argmax(metrics["val_deam_ccc_epoch"]))
        best_mtg_idx = int(np.argmax(metrics["mtg_f1_micro"]))

        record: dict[str, float | str] = {
            "label": label,
            "run_dir": str(run_dir),
            "best_val_epoch": int(metrics["epoch"][best_val_idx]),
            "best_val_loss": float(metrics["val_loss"][best_val_idx]),
            "best_val_deam_ccc_epoch": float(metrics["val_deam_ccc_epoch"][best_val_idx]),
            "best_val_mtg_f1_micro": float(metrics["mtg_f1_micro"][best_val_idx]),
            "best_deam_epoch": int(metrics["epoch"][best_deam_idx]),
            "best_deam_ccc_epoch": float(metrics["val_deam_ccc_epoch"][best_deam_idx]),
            "best_mtg_epoch": int(metrics["epoch"][best_mtg_idx]),
            "best_mtg_f1_micro": float(metrics["mtg_f1_micro"][best_mtg_idx]),
        }
        record.update(test_metrics)
        records.append(record)
        histories[label] = metrics

    return records, histories


def _write_summary_csv(records: list[dict[str, float | str]], out_path: Path) -> None:
    fieldnames = [
        "label",
        "run_dir",
        "best_val_epoch",
        "best_val_loss",
        "best_val_deam_ccc_epoch",
        "best_val_mtg_f1_micro",
        "best_deam_epoch",
        "best_deam_ccc_epoch",
        "best_mtg_epoch",
        "best_mtg_f1_micro",
        "test_loss",
        "test_deam_ccc_epoch",
        "test_deam_valence",
        "test_deam_arousal",
        "test_deam_rmse",
        "test_deam_pearson",
        "test_mtg_threshold",
        "test_mtg_roc_auc_micro",
        "test_mtg_roc_auc_macro",
        "test_mtg_pr_auc_micro",
        "test_mtg_pr_auc_macro",
        "test_mtg_f1_micro",
        "test_mtg_f1_macro",
        "test_mtg_precision_micro",
        "test_mtg_precision_macro",
        "test_mtg_recall_micro",
        "test_mtg_recall_macro",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_test_comparison(records: list[dict[str, float | str]], out_path: Path) -> None:
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Main Experiment: Early vs Late Fusion", fontsize=14, fontweight="bold")

    deam_ax = axes[0]
    deam_metrics = [
        ("test_deam_ccc_epoch", "CCC"),
        ("test_deam_valence", "Valence"),
        ("test_deam_arousal", "Arousal"),
        ("test_deam_pearson", "Pearson"),
        ("test_deam_rmse", "RMSE"),
    ]
    deam_x = np.arange(len(deam_metrics))
    for idx, record in enumerate(records):
        values = [float(record[key]) for key, _ in deam_metrics]
        bars = deam_ax.bar(
            deam_x + (idx - 0.5) * width,
            values,
            width=width,
            label=str(record["label"]),
            color=COLORS[str(record["label"])],
        )
        _add_bar_labels(deam_ax, bars)

    deam_ax.set_xticks(deam_x)
    deam_ax.set_xticklabels([name for _, name in deam_metrics])
    deam_ax.set_ylabel("Score")
    deam_ax.set_ylim(0.0, 0.9)
    deam_ax.set_title("(a) DEAM Test Metrics")
    deam_ax.grid(True, axis="y", alpha=0.3)
    deam_ax.annotate("RMSE lower is better", xy=(0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=8)
    deam_ax.legend(loc="upper left", fontsize=9)

    mtg_ax = axes[1]
    mtg_metrics = [
        ("test_mtg_roc_auc_micro", "ROC-AUC μ"),
        ("test_mtg_roc_auc_macro", "ROC-AUC M"),
        ("test_mtg_pr_auc_micro", "PR-AUC μ"),
        ("test_mtg_f1_micro", "F1 μ"),
    ]
    mtg_x = np.arange(len(mtg_metrics))
    for idx, record in enumerate(records):
        values = [float(record[key]) for key, _ in mtg_metrics]
        bars = mtg_ax.bar(
            mtg_x + (idx - 0.5) * width,
            values,
            width=width,
            label=str(record["label"]),
            color=COLORS[str(record["label"])],
        )
        _add_bar_labels(mtg_ax, bars)

    mtg_ax.set_xticks(mtg_x)
    mtg_ax.set_xticklabels([name for _, name in mtg_metrics])
    mtg_ax.set_ylabel("Score")
    mtg_ax.set_ylim(0.0, 0.9)
    mtg_ax.set_title("(b) MTG Test Metrics")
    mtg_ax.grid(True, axis="y", alpha=0.3)
    mtg_ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    _save(fig, out_path)


def plot_validation_curves(histories: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Main Experiment Validation Curves", fontsize=14, fontweight="bold")

    panels = [
        ("val_loss", "Validation Loss", False),
        ("val_deam_ccc_epoch", "DEAM CCC (epoch-level)", True),
        ("mtg_f1_micro", "MTG F1-micro", True),
        ("mtg_roc_auc_micro", "MTG ROC-AUC micro", True),
    ]

    for ax, (key, ylabel, higher_is_better) in zip(axes.flat, panels):
        for label, data in histories.items():
            color = COLORS[label]
            ax.plot(data["epoch"], data[key], label=label, color=color, linewidth=2.0)
            values = data[key]
            best_idx = int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
            ax.scatter(data["epoch"][best_idx], values[best_idx], color=color, s=30, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=9)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")

    fig.tight_layout()
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw-data figures for the main experiments.")
    parser.add_argument(
        "--out_dir",
        default="thesis_figures/main_experiment_20260308",
        help="Output directory for figures and summary CSV",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    records, histories = _load_experiments()
    _write_summary_csv(records, out_dir / "main_experiment_summary.csv")
    plot_test_comparison(records, out_dir / "main_experiment_test_comparison")
    plot_validation_curves(histories, out_dir / "main_experiment_validation_curves")

    print(f"Saved figures and summary to: {out_dir}")


if __name__ == "__main__":
    main()
