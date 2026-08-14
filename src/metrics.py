"""Binary classification metrics and saved diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    true_labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    true_labels = np.asarray(true_labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    roc_auc = (
        roc_auc_score(true_labels, probabilities)
        if len(np.unique(true_labels)) == 2
        else float("nan")
    )
    return {
        "n": int(len(true_labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true_labels, predicted)),
        "precision": float(precision_score(true_labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(true_labels, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(true_labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_labels": predicted.tolist(),
    }


def plot_training_history(history: dict[str, list[float]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, title, ylabel in (
        ("loss", "training_validation_loss.png", "Training vs Validation Loss", "Binary crossentropy"),
        ("accuracy", "training_validation_accuracy.png", "Training vs Validation Accuracy", "Accuracy"),
    ):
        if metric not in history or f"val_{metric}" not in history:
            continue
        epochs = np.arange(1, len(history[metric]) + 1)
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.plot(epochs, history[metric], marker="o", label="Training")
        axis.plot(epochs, history[f"val_{metric}"], marker="o", label="Validation")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)


def plot_roc(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    roc_auc: float,
    output_path: Path,
    title: str = "Locked Test ROC Curve",
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, probabilities)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(
        title=title,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Sensitivity)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(
    metrics: dict[str, Any],
    output_path: Path,
    class_names: tuple[str, str] = ("Normal", "Cataract"),
    title: str = "Locked Test Confusion Matrix",
) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_totals, where=row_totals != 0) * 100.0
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], labels=list(class_names))
    axis.set_yticks([0, 1], labels=list(class_names))
    axis.set(xlabel="Predicted class", ylabel="True class", title=title)
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{percentages[row, column]:.1f}%",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
