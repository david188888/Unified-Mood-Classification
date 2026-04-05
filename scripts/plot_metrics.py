#!/usr/bin/env python
"""Plot epoch-level metrics from metrics.csv into publication-ready figures."""

import argparse
import csv
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

SCIENTIFIC_COLORS = {
    "train": "#2878B5",
    "val": "#C82423",
    "ccc": "#2878B5",
    "valence": "#9AC9DB",
    "arousal": "#C82423",
    "pearson": "#F8AC8C",
    "rmse": "#FFBE7A",
    "f1": "#2878B5",
    "pr": "#FFBE7A",
    "roc": "#59A14F",
}


def _read_metrics(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    data: Dict[str, list] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                data[key].append(np.nan)
                continue
            if key == "epoch":
                try:
                    data[key].append(int(float(value)))
                except ValueError:
                    data[key].append(np.nan)
            else:
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(np.nan)

    return {key: np.asarray(values) for key, values in data.items()}


def _save_figure(fig: plt.Figure, out_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def _plot_loss(data: Dict[str, np.ndarray]) -> plt.Figure:
    epochs = data["epoch"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(epochs, data.get("train_loss", np.array([])), label="Train Loss", color=SCIENTIFIC_COLORS["train"], linewidth=2.0)
    ax.plot(epochs, data.get("val_loss", np.array([])), label="Val Loss", color=SCIENTIFIC_COLORS["val"], linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def _plot_deam(data: Dict[str, np.ndarray]) -> plt.Figure:
    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True)

    ax = axes[0]
    series = [
        ("val_deam_ccc_epoch", "CCC (mean)", SCIENTIFIC_COLORS["ccc"]),
        ("val_deam_ccc_valence", "CCC (valence)", SCIENTIFIC_COLORS["valence"]),
        ("val_deam_ccc_arousal", "CCC (arousal)", SCIENTIFIC_COLORS["arousal"]),
        ("val_deam_pearson", "Pearson", SCIENTIFIC_COLORS["pearson"]),
    ]
    for key, label, color in series:
        if key in data:
            ax.plot(epochs, data[key], label=label, color=color, linewidth=2.0)
    ax.set_ylabel("Correlation")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "val_deam_rmse" in data:
        ax.plot(epochs, data["val_deam_rmse"], label="RMSE", color=SCIENTIFIC_COLORS["rmse"], linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)

    return fig


def _plot_mtg(data: Dict[str, np.ndarray]) -> plt.Figure:
    epochs = data["epoch"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = [
        ("mtg_f1_micro", "F1 micro", SCIENTIFIC_COLORS["f1"]),
        ("mtg_pr_auc_micro", "PR-AUC micro", SCIENTIFIC_COLORS["pr"]),
        ("mtg_roc_auc_micro", "ROC-AUC micro", SCIENTIFIC_COLORS["roc"]),
    ]
    for key, label, color in series:
        if key in data:
            ax.plot(epochs, data[key], label=label, color=color, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics for paper-ready figures.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--out_dir", default=None, help="Output directory for figures")
    args = parser.parse_args()

    data = _read_metrics(args.metrics)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.metrics), "figures")
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10})

    _save_figure(_plot_loss(data), out_dir, "loss_curve")
    _save_figure(_plot_deam(data), out_dir, "deam_metrics")
    _save_figure(_plot_mtg(data), out_dir, "mtg_metrics")


if __name__ == "__main__":
    main()
