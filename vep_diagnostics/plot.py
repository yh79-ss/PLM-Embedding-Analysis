"""Optional compact plots of emitted summary tables; no recomputation."""
from pathlib import Path

import numpy as np
import pandas as pd


def plot_summary(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(args.summary)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if "task" not in frame or frame.empty:
        raise ValueError("Expected a nonempty summary table with task column")
    label = frame.task.astype(str)
    if "model" in frame:
        label = label + " / " + frame.model.astype(str)
    values = [c for c in ("delta", "oracle_gap", "mean_weight_cosine", "mean_rho", "median_reference_percentile", "mean_substitution_jsd") if c in frame]
    if not values:
        raise ValueError("Unsupported summary; use probe, diagnose, cosine, support, compare, shift, or composition summary.csv")
    value = values[0]
    if not np.isfinite(frame[value]).any():
        raise ValueError("No finite summary values to plot")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(frame))))
    positions = np.arange(len(frame))
    ax.scatter(frame[value], positions, color="#176b87")
    if value == "delta":
        ax.hlines(positions, frame.ci_low, frame.ci_high, color="#176b87")
        ax.axvline(0, color="gray", linewidth=0.8)
    ax.set(yticks=positions, yticklabels=label, xlabel=value.replace("_", " "))
    ax.invert_yaxis()
    caption = " | ".join(str(frame[c].iloc[0]) for c in ("analysis_scope", "label_protocol") if c in frame)
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(caption, 95)), fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
