"""
Curriculum Stage Progression Plot

Shows success rate and collision rate across Stages 1-6 from curriculum_eval.py,
with gate success rate overlaid as a secondary line.
Error bars show ±1 std of the gate evaluation history per stage.

Data sources:
  - ./models/curriculum_eval_summary.csv  (from curriculum_eval.py)
  - ./logs/stage_X/stage_gate_log.csv     (gate evaluation history per stage)

Usage:
    python plot_curriculum_stage_progression.py
    python plot_curriculum_stage_progression.py --out_dir ./figures --show
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv",  type=str,
                   default="./models/curriculum_eval_summary.csv")
    p.add_argument("--log_dir",   type=str, default="./logs")
    p.add_argument("--out_dir",   type=str, default="./figures")
    p.add_argument("--stages",    type=str, default="1,2,3,4,5,6",
                   help="Comma-separated stage IDs to include")
    p.add_argument("--show",      action="store_true")
    return p.parse_args()


def load_gate_stats(log_dir: str, stage_id: int):
    """
    Load stage_gate_log.csv and return (mean_success, std_success)
    across all gate evaluations for that stage.
    Returns (None, None) if file not found.
    """
    path = os.path.join(log_dir, f"stage_{stage_id}", "stage_gate_log.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if df.empty or "success_rate" not in df.columns:
        return None, None
    return float(df["success_rate"].mean()), float(df["success_rate"].std(ddof=0))


def main():
    args = parse_args()
    stages = [int(s) for s in args.stages.split(",")]

    # --- Load eval summary ---
    if not os.path.exists(args.eval_csv):
        raise FileNotFoundError(
            f"Eval CSV not found: {args.eval_csv}\n"
            "Run: python curriculum_eval.py --stage_start 1 --stage_end 6 --episodes 100"
        )
    eval_df = pd.read_csv(args.eval_csv)
    eval_df = eval_df[eval_df["stage_id"].isin(stages)].sort_values("stage_id")

    x          = eval_df["stage_id"].values
    success    = eval_df["success_rate"].values
    collision  = eval_df["collision_rate"].values

    # --- Load gate stats per stage ---
    gate_mean = []
    gate_std  = []
    for sid in x:
        m, s = load_gate_stats(args.log_dir, sid)
        gate_mean.append(m)
        gate_std.append(s if s is not None else 0.0)

    gate_mean = np.array([v if v is not None else np.nan for v in gate_mean])
    gate_std  = np.array(gate_std)

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: success and collision bars
    bar_w = 0.3
    x_pos = np.arange(len(x))

    bars1 = ax1.bar(x_pos - bar_w/2, success,   bar_w,
                    label="Eval Success Rate",
                    color="#43A047", alpha=0.85, edgecolor="white", zorder=3)
    bars2 = ax1.bar(x_pos + bar_w/2, collision, bar_w,
                    label="Eval Collision Rate",
                    color="#E53935", alpha=0.85, edgecolor="white", zorder=3)

    # Value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                     f"{h:.2f}", ha="center", va="bottom",
                     fontsize=8, color="#333")

    # Secondary line: gate success rate with ±1 std error bars
    valid = ~np.isnan(gate_mean)
    if valid.any():
        ax1.errorbar(
            x_pos[valid], gate_mean[valid],
            yerr=gate_std[valid],
            fmt="D--", color="#1565C0", linewidth=2.0,
            markersize=8, capsize=5, capthick=1.5,
            label="Gate Success Rate (mean ± 1 std)",
            zorder=5
        )

    # Formatting
    ax1.set_xlabel("Curriculum Stage", fontsize=12)
    ax1.set_ylabel("Rate", fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"Stage {s}" for s in x], fontsize=10)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Stage difficulty annotation on x-axis
    stage_info = {
        1: "10m\n2S/1D",
        2: "20m\n2S/2D",
        3: "30m\n2S/3D",
        4: "40m\n3S/3D",
        5: "50m\n3S/4D",
        6: "50m\n4S/5D\n+orbital",
    }
    for i, sid in enumerate(x):
        if sid in stage_info:
            ax1.text(x_pos[i], -0.13, stage_info[sid],
                     ha="center", va="top", fontsize=7.5,
                     color="#555", transform=ax1.get_xaxis_transform())

    ax1.set_title(
        "Curriculum Stage Progression: Success & Collision Rate\n"
        "with Gate Evaluation History (±1 std)",
        fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="lower left", fontsize=9, frameon=True)

    # Highlight Stage 6 (orbital motion introduced)
    if 6 in x:
        idx6 = list(x).index(6)
        ax1.axvspan(idx6 - 0.5, idx6 + 0.5, alpha=0.06,
                    color="#1565C0", zorder=0)
        ax1.text(idx6, 1.10, "Orbital\nmotion\nadded",
                 ha="center", va="top", fontsize=7.5,
                 color="#1565C0", style="italic")

    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"curriculum_stage_progression.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
