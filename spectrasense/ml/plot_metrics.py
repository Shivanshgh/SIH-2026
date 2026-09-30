"""Render a presentation-ready PNG from modulation_model_metrics.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "models" / "modulation_model_metrics.json"
OUTPUT_PATH = ROOT / "reports" / "modulation_model_evaluation.png"


def main():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.15, 1.25, 1]})
    fig.patch.set_facecolor("#0b1220")
    for ax in axes:
        ax.set_facecolor("#111c2e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#334155", alpha=0.45)
        ax.set_axisbelow(True)

    # Prefer public corpus results when present; otherwise show synthetic holdout.
    radio_available = "radioml2016_10a_accuracy_by_snr_db" in metrics
    snr = metrics.get("radioml2016_10a_accuracy_by_snr_db", metrics.get("accuracy_by_snr_db", {}))
    x = sorted((int(level), float(score)) for level, score in snr.items())
    axes[0].plot([p[0] for p in x], [100 * p[1] for p in x], color="#38bdf8", marker="o", lw=2.5)
    axes[0].set(title="{} holdout by SNR".format("RadioML" if radio_available else "Synthetic"), xlabel="SNR (dB)", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0].set_xticks([p[0] for p in x][::2])
    axes[0].tick_params(axis="x", rotation=35)

    # Remove unsupported classes from older four-class matrices.
    matrix = metrics.get("radioml2016_10a_confusion_matrix", metrics.get("confusion_matrix", []))
    labels = metrics.get("radioml2016_10a_classes", metrics.get("classes", []))
    keep = [i for i, row in enumerate(matrix) if sum(row) > 0]
    labels = [labels[i] for i in keep]
    matrix = np.asarray([[matrix[i][j] for j in keep] for i in keep])
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set(title="RadioML confusion matrix", xlabel="Predicted", ylabel="Actual",
                xticks=range(len(labels)), xticklabels=labels,
                yticks=range(len(labels)), yticklabels=labels)
    axes[1].tick_params(axis="x", rotation=35)
    for row in range(len(labels)):
        for col in range(len(labels)):
            axes[1].text(col, row, str(matrix[row, col]), ha="center", va="center",
                         color="white" if matrix[row, col] > matrix.max() * 0.5 else "#0b1220", fontsize=10)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].grid(False)

    names = ["Synthetic baseline", "Final model"]
    values = [100 * metrics["synthetic_only_baseline_accuracy"],
              100 * metrics["combined_model_synthetic_holdout_accuracy"]]
    bars = axes[2].bar(names, values, color=["#64748b", "#a78bfa"], width=0.58)
    axes[2].set(title="Synthetic holdout", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[2].tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", color="white")

    fig.suptitle("SpectraSense | Model evaluation", color="white", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025,
             "Held-out simulated signals only • Public data included: {} • Not field-performance estimates".format("RadioML" if radio_available else "no"),
             ha="center", color="#94a3b8", fontsize=9)
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
