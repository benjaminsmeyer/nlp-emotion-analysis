"""
compare_from_summaries.py

Reads results/{model}/summary.txt for all available models and generates
three comparison charts saved to results/comparison/.

Usage:
    python results/compare_from_summaries.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]
RESULTS_DIR = Path(__file__).parent
COMPARISON_DIR = RESULTS_DIR / "comparison"
COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_summary(path: Path) -> dict | None:
    """Parse a summary.txt file and return {model, accuracy, precision, recall, f1}."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None

    metrics = {}
    in_metrics = False
    for line in lines:
        if line.startswith("Model:"):
            metrics["model"] = line.split(":", 1)[1].strip()
        elif line.strip() == "Test Metrics:":
            in_metrics = True
        elif in_metrics and " : " in line:
            key, val = line.strip().split(" : ", 1)
            key = key.strip()
            if key in METRIC_NAMES:
                metrics[key] = float(val.strip())

    if all(k in metrics for k in ["model"] + METRIC_NAMES):
        return metrics
    return None


def load_all_summaries() -> list[dict]:
    summaries = []
    for summary_path in sorted(RESULTS_DIR.glob("*/summary.txt")):
        data = parse_summary(summary_path)
        if data:
            summaries.append(data)
            print(f"Loaded: {summary_path}")
        else:
            print(f"Skipped (incomplete): {summary_path}")
    return summaries


def save_grouped_bar(summaries: list[dict]) -> None:
    """Grouped bar chart: 4 metric bars clustered per model."""
    models = [s["model"] for s in summaries]
    n_models = len(models)
    n_metrics = len(METRIC_NAMES)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, color) in enumerate(zip(METRIC_NAMES, COLORS)):
        values = [s[metric] for s in summaries]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("All Metrics by Model")
    ax.legend(loc="lower right")
    ax.xaxis.grid(False)
    fig.tight_layout()

    _save_fig(fig, COMPARISON_DIR / "metrics_by_model.png")


def save_per_metric_subplots(summaries: list[dict]) -> None:
    """2×2 subplot grid: one horizontal bar chart per metric, sorted best→worst."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, metric, color in zip(axes, METRIC_NAMES, COLORS):
        sorted_summaries = sorted(summaries, key=lambda s: s[metric])
        models = [s["model"] for s in sorted_summaries]
        values = [s[metric] for s in sorted_summaries]

        bars = ax.barh(models, values, color=color)
        ax.set_xlim(0, 1.05)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Score")
        ax.yaxis.grid(False)

        for bar, val in zip(bars, values):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=8,
            )

    fig.suptitle("Per-Metric Comparison Across Models", fontsize=15)
    fig.tight_layout()

    _save_fig(fig, COMPARISON_DIR / "per_metric_comparison.png")


def save_radar_chart(summaries: list[dict]) -> None:
    """Radar chart with one polygon per model over accuracy/precision/recall/F1 axes."""
    n_metrics = len(METRIC_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for summary, color in zip(summaries, COLORS):
        values = [summary[m] for m in METRIC_NAMES]
        values += values[:1]  # close the polygon
        ax.plot(angles, values, color=color, linewidth=2, label=summary["model"])
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_NAMES)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    _save_fig(fig, COMPARISON_DIR / "radar_comparison.png")


def main() -> None:
    summaries = load_all_summaries()
    if len(summaries) < 2:
        print("Need at least 2 models with summary.txt to generate comparisons.")
        return

    save_grouped_bar(summaries)
    save_per_metric_subplots(summaries)
    save_radar_chart(summaries)
    print(f"\nDone. Charts saved to {COMPARISON_DIR}/")


if __name__ == "__main__":
    main()
