from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize":  14,
    "axes.labelsize":  11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.figsize":  (8, 6),
})

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]

BAR_COLOR = "#4878CF"  # single reusable bar color


def ensure_results_dirs(model_name: str) -> None:
    Path(f"results/{model_name}").mkdir(parents=True, exist_ok=True)


def _save_fig(fig, out_path: str, bbox_inches="tight") -> None:
    fig.savefig(out_path, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _annotate_bars(ax, bars, labels, offset=0.005, fontsize=9) -> None:
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                label, ha='center', va='bottom', fontsize=fontsize)


def save_confusion_matrix(y_true, y_pred, model_name: str, label_names: dict, f1_score: float = None) -> None:
    ensure_results_dirs(model_name)

    sorted_keys = sorted(label_names)
    labels = [label_names[i] for i in sorted_keys]
    matrix = sk_confusion_matrix(y_true, y_pred, labels=sorted_keys)

    row_sums = matrix.sum(axis=1, keepdims=True)
    pct_matrix = matrix / np.where(row_sums == 0, 1, row_sums) * 100
    annot_array = np.array([[f"{p:.1f}%" for p in row] for row in pct_matrix])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=annot_array, fmt="", xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.tick_params(labelsize=11)

    _save_fig(fig, f"results/{model_name}/confusion_matrix.png")


def save_metrics_comparison(results_dict: dict, models_run: list) -> None:
    avg_values = [np.mean([results_dict[m].get(metric, 0) for m in models_run]) for metric in METRIC_NAMES]

    fig, ax = plt.subplots()
    bars = ax.bar(METRIC_NAMES, avg_values, color=BAR_COLOR)
    _annotate_bars(ax, bars, [f"{v:.2f}" for v in avg_values])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics — All Models")
    ax.xaxis.grid(False)
    fig.tight_layout()

    _save_fig(fig, "results/metrics_comparison.png")


def _setup_time_ax(ax, x, models_run: list, values: list, title: str, fmt) -> None:
    bars = ax.bar(x, values, 0.5, color=BAR_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(models_run, rotation=15, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    ax.xaxis.grid(False)
    nonzero = [(b, v) for b, v in zip(bars, values) if v > 0]
    if nonzero:
        nz_bars, nz_vals = zip(*nonzero)
        _annotate_bars(ax, nz_bars, [fmt(v) for v in nz_vals])


def save_runtime_comparison(results_dict: dict, models_run: list) -> None:
    train_times = [results_dict[m].get("training_time_seconds", 0) for m in models_run]
    infer_times = [results_dict[m].get("inference_time_per_sample", 0) for m in models_run]

    x = np.arange(len(models_run))
    fig, (ax_train, ax_infer) = plt.subplots(1, 2, figsize=(12, 5))
    _setup_time_ax(ax_train, x, models_run, train_times, "Training Time", lambda v: f"{v:.1f}s")
    _setup_time_ax(ax_infer, x, models_run, infer_times, "Inference Time per Sample", lambda v: f"{v * 1000:.2f}ms")
    fig.suptitle("Training and Inference Time by Model")
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.95, wspace=0.35)
    _save_fig(fig, "results/runtime_comparison.png", bbox_inches=None)


def save_training_curves(history: dict, model_name: str) -> None:
    ensure_results_dirs(model_name)

    epochs = history["epoch"]
    val_loss = history["val_loss"]
    val_accuracy = history["val_accuracy"]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss")
    ax_loss.plot(epochs, val_loss, label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{model_name} — Loss Curves")
    ax_loss.legend()

    # Accuracy subplot
    ax_acc.plot(epochs, val_accuracy, label="Val Accuracy", color="green")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{model_name} — Validation Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    _save_fig(fig, f"results/{model_name}/training_curves.png")


def save_model_metrics_bar(metrics: dict, model_name: str) -> None:
    ensure_results_dirs(model_name)

    values = [metrics[m] for m in METRIC_NAMES]

    fig, ax = plt.subplots()
    bars = ax.bar(METRIC_NAMES, values, color=BAR_COLOR)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Test Metrics")
    ax.xaxis.grid(False)
    _annotate_bars(ax, bars, [f"{v:.2f}" for v in values])
    fig.tight_layout()

    _save_fig(fig, f"results/{model_name}/metrics.png")


def save_all_models_comparison(results_dict: dict) -> None:
    ensure_results_dirs("comparison")

    # barh renders bottom-to-top, so sort ascending → best model ends up at top
    display_order = sorted(results_dict, key=lambda m: results_dict[m]['f1'])

    f1_values = [results_dict[m]['f1'] for m in display_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(display_order, f1_values, color=BAR_COLOR)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F1")
    ax.set_title("Macro F1-Score by Model")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    fig.tight_layout()

    _save_fig(fig, "results/comparison/all_models_comparison.png")


def save_text_summary(metrics: dict, hyperparams: dict, model_name: str) -> None:
    """Write a plain-text summary of hyperparameters and test metrics."""
    ensure_results_dirs(model_name)

    out_path = f"results/{model_name}/summary.txt"
    with open(out_path, "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key} : {value}\n")
        f.write("\nTest Metrics:\n")
        for name in METRIC_NAMES:
            f.write(f"  {name:<9} : {metrics[name]:.4f}\n")

    print(f"Saved: {out_path}")
