#!/usr/bin/env python3
"""Generate publication-ready figures for feature ablation results."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = [
    ("early_ablation_mert_only", "MERT"),
    ("early_ablation_mert_mel", "MERT+Mel"),
    ("early_ablation_mert_mel_chroma", "MERT+Mel+Chroma"),
    ("early_ablation_full", "MERT+Mel+Chroma+Tempogram"),
]

COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
METRIC_COLORS = {
    "ccc": "#2878B5",
    "valence": "#9AC9DB",
    "arousal": "#C82423",
    "pearson": "#F8AC8C",
    "rmse": "#FFBE7A",
    "f1": "#2878B5",
    "pr": "#FFBE7A",
    "roc": "#59A14F",
    "precision": "#C82423",
    "recall": "#76B7B2",
}
COLORS = ["#9AC9DB", "#5F97C6", "#2F5C85", "#C82423"]

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


def _load_suite(suite_dir: Path) -> tuple[list[dict[str, float | str]], dict[str, dict[str, np.ndarray]]]:
    records: list[dict[str, float | str]] = []
    histories: dict[str, dict[str, np.ndarray]] = {}

    for exp_name, label in EXPERIMENTS:
        run_dir = suite_dir / exp_name / "unified_mood_model_early"
        metrics_path = run_dir / "metrics.csv"
        log_path = run_dir / "training.log"

        metrics = _read_metrics(metrics_path)
        test_metrics = _parse_test_metrics(log_path)

        best_val_idx = int(np.argmin(metrics["val_loss"]))
        best_mtg_idx = int(np.argmax(metrics["mtg_f1_micro"]))

        record: dict[str, float | str] = {
            "experiment": exp_name,
            "label": label,
            "best_val_epoch": int(metrics["epoch"][best_val_idx]),
            "best_val_loss": float(metrics["val_loss"][best_val_idx]),
            "best_val_deam_ccc_epoch": float(metrics["val_deam_ccc_epoch"][best_val_idx]),
            "best_val_mtg_f1_micro": float(metrics["mtg_f1_micro"][best_val_idx]),
            "best_mtg_epoch": int(metrics["epoch"][best_mtg_idx]),
            "best_mtg_f1_micro": float(metrics["mtg_f1_micro"][best_mtg_idx]),
            "best_mtg_roc_auc_micro": float(metrics["mtg_roc_auc_micro"][best_mtg_idx]),
        }
        record.update(test_metrics)
        records.append(record)
        histories[label] = metrics

    return records, histories


def _write_summary_csv(records: list[dict[str, float | str]], out_path: Path) -> None:
    fieldnames = [
        "experiment",
        "label",
        "best_val_epoch",
        "best_val_loss",
        "best_val_deam_ccc_epoch",
        "best_val_mtg_f1_micro",
        "best_mtg_epoch",
        "best_mtg_f1_micro",
        "best_mtg_roc_auc_micro",
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


def plot_validation_curves(histories: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Feature Ablation Validation Curves", fontsize=14, fontweight="bold")

    panels = [
        ("val_deam_ccc_epoch", "DEAM CCC (epoch-level)", True),
        ("val_deam_rmse", "DEAM RMSE", False),
        ("mtg_f1_micro", "MTG F1-micro", True),
        ("mtg_roc_auc_micro", "MTG ROC-AUC micro", True),
    ]

    for ax, (key, ylabel, higher_is_better) in zip(axes.flat, panels):
        for color, (label, data) in zip(COLORS, histories.items()):
            ax.plot(data["epoch"], data[key], label=label, color=color, linewidth=2.0)
            values = data[key]
            best_idx = int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
            ax.scatter(
                data["epoch"][best_idx],
                values[best_idx],
                color=color,
                s=25,
                zorder=3,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(ncol=2, fontsize=9, frameon=True)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")

    fig.tight_layout()
    _save(fig, out_path)


def plot_deam_test_bars(records: list[dict[str, float | str]], out_path: Path) -> None:
    labels = [str(record["label"]) for record in records]
    x = np.arange(len(labels))
    width = 0.18

    metrics = [
        ("test_deam_ccc_epoch", "CCC (overall)"),
        ("test_deam_valence", "CCC (valence)"),
        ("test_deam_arousal", "CCC (arousal)"),
        ("test_deam_pearson", "Pearson r"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, (key, name) in enumerate(metrics):
        values = [float(record[key]) for record in records]
        color = [
            METRIC_COLORS["ccc"],
            METRIC_COLORS["valence"],
            METRIC_COLORS["arousal"],
            METRIC_COLORS["pearson"],
        ][idx]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, max(float(record["test_deam_arousal"]) for record in records) + 0.1)
    ax.set_title("Feature Ablation: DEAM Test Metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    rmse_ax = ax.twinx()
    rmse_ax.plot(
        x,
        [float(record["test_deam_rmse"]) for record in records],
        color=METRIC_COLORS["rmse"],
        marker="o",
        linewidth=2.0,
        label="RMSE",
    )
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.set_ylim(
        min(float(record["test_deam_rmse"]) for record in records) - 0.05,
        max(float(record["test_deam_rmse"]) for record in records) + 0.05,
    )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = rmse_ax.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, ncol=3, fontsize=9, loc="upper left")

    fig.tight_layout()
    _save(fig, out_path)


def plot_mtg_test_bars(records: list[dict[str, float | str]], out_path: Path) -> None:
    labels = [str(record["label"]) for record in records]
    x = np.arange(len(labels))
    width = 0.16

    metrics = [
        ("test_mtg_f1_micro", "F1-micro"),
        ("test_mtg_pr_auc_micro", "PR-AUC micro"),
        ("test_mtg_roc_auc_micro", "ROC-AUC micro"),
        ("test_mtg_precision_micro", "Precision micro"),
        ("test_mtg_recall_micro", "Recall micro"),
    ]

    fig, ax = plt.subplots(figsize=(13, 5))

    for idx, (key, name) in enumerate(metrics):
        values = [float(record[key]) for record in records]
        color = [
            METRIC_COLORS["f1"],
            METRIC_COLORS["pr"],
            METRIC_COLORS["roc"],
            METRIC_COLORS["precision"],
            METRIC_COLORS["recall"],
        ][idx]
        ax.bar(x + (idx - 2) * width, values, width=width, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 0.85)
    ax.set_title("Feature Ablation: MTG Test Metrics")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=9)

    fig.tight_layout()
    _save(fig, out_path)


def plot_final_comparison(records: list[dict[str, float | str]], out_path: Path) -> None:
    labels = [str(record["label"]) for record in records]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Ablation Final Test Comparison", fontsize=14, fontweight="bold")

    deam_ax = axes[0]
    width = 0.18
    deam_metrics = [
        ("test_deam_ccc_epoch", "CCC"),
        ("test_deam_valence", "Valence"),
        ("test_deam_arousal", "Arousal"),
        ("test_deam_pearson", "Pearson"),
    ]
    for idx, (key, name) in enumerate(deam_metrics):
        values = [float(record[key]) for record in records]
        color = [
            METRIC_COLORS["ccc"],
            METRIC_COLORS["valence"],
            METRIC_COLORS["arousal"],
            METRIC_COLORS["pearson"],
        ][idx]
        deam_ax.bar(x + (idx - 1.5) * width, values, width=width, label=name, color=color)

    deam_ax.set_xticks(x)
    deam_ax.set_xticklabels(labels, rotation=12, ha="right")
    deam_ax.set_ylabel("Score")
    deam_ax.set_ylim(0.0, 0.9)
    deam_ax.set_title("(a) DEAM Metrics")
    deam_ax.grid(True, axis="y", alpha=0.3)

    rmse_ax = deam_ax.twinx()
    rmse_ax.plot(
        x,
        [float(record["test_deam_rmse"]) for record in records],
        color=METRIC_COLORS["rmse"],
        marker="o",
        linewidth=2.0,
        label="RMSE",
    )
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.set_ylim(
        min(float(record["test_deam_rmse"]) for record in records) - 0.05,
        max(float(record["test_deam_rmse"]) for record in records) + 0.05,
    )
    deam_ax.annotate("RMSE lower is better", xy=(0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=8)
    handles1, labels1 = deam_ax.get_legend_handles_labels()
    handles2, labels2 = rmse_ax.get_legend_handles_labels()
    deam_ax.legend(handles1 + handles2, labels1 + labels2, ncol=3, loc="upper left", fontsize=8)

    mtg_ax = axes[1]
    mtg_metrics = [
        ("test_mtg_f1_micro", "F1 μ"),
        ("test_mtg_pr_auc_micro", "PR-AUC μ"),
        ("test_mtg_roc_auc_micro", "ROC-AUC μ"),
        ("test_mtg_precision_micro", "Precision μ"),
        ("test_mtg_recall_micro", "Recall μ"),
    ]
    width = 0.15
    for idx, (key, name) in enumerate(mtg_metrics):
        values = [float(record[key]) for record in records]
        color = [
            METRIC_COLORS["f1"],
            METRIC_COLORS["pr"],
            METRIC_COLORS["roc"],
            METRIC_COLORS["precision"],
            METRIC_COLORS["recall"],
        ][idx]
        mtg_ax.bar(x + (idx - 2) * width, values, width=width, label=name, color=color)

    mtg_ax.set_xticks(x)
    mtg_ax.set_xticklabels(labels, rotation=12, ha="right")
    mtg_ax.set_ylabel("Score")
    mtg_ax.set_ylim(0.0, 0.85)
    mtg_ax.set_title("(b) MTG Metrics")
    mtg_ax.grid(True, axis="y", alpha=0.3)
    mtg_ax.legend(loc="upper left", ncol=2, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw-data feature ablation figures.")
    parser.add_argument(
        "--suite_dir",
        default="runs/feature_ablation_20260331_001622",
        help="Path to the feature ablation suite directory",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for figures and summary CSV",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path("thesis_figures") / suite_dir.name
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

    records, histories = _load_suite(suite_dir)
    _write_summary_csv(records, out_dir / "feature_ablation_summary.csv")
    plot_final_comparison(records, out_dir / "feature_ablation_final_comparison")
    plot_validation_curves(histories, out_dir / "feature_ablation_validation_curves")
    plot_deam_test_bars(records, out_dir / "feature_ablation_deam_test")
    plot_mtg_test_bars(records, out_dir / "feature_ablation_mtg_test")

    print(f"Saved figures and summary to: {out_dir}")


if __name__ == "__main__":
    main()
