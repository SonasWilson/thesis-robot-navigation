"""
Plot SAC vs TD3 unseen generalization results.

Reads the CSV produced by eval_sac_vs_td3.py and generates:
  1. Grouped bar chart — success rate per condition
  2. Grouped bar chart — collision rate per condition
  3. Grouped bar chart — average reward per condition
  4. Radar chart       — overall profile across all conditions
  5. Summary bar       — aggregate mean across all conditions

Usage:
    python plot_sac_vs_td3.py --csv sac_vs_td3_unseen.csv --out_dir ./plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
SAC_COLOR = "#2196F3"   # blue
TD3_COLOR = "#F44336"   # red
BAR_WIDTH  = 0.35
FIGSIZE_BAR   = (10, 5)
FIGSIZE_RADAR = (6, 6)
FIGSIZE_AGG   = (6, 4)

CONDITION_LABELS = {
    "UA": "UA\nMore Obstacles",
    "UB": "UB\nLarger Arena",
    "UC": "UC\nFaster Dynamics",
    "UD": "UD\nShifted Positions",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     type=str, default="sac_vs_td3_unseen.csv")
    p.add_argument("--out_dir", type=str, default="./plots")
    p.add_argument("--show",    action="store_true", help="Open interactive windows")
    return p.parse_args()


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["condition_label"] = df["condition"].map(
        lambda c: CONDITION_LABELS.get(c, c))
    return df


def _bar_group(ax, conditions, sac_vals, td3_vals, ylabel, title, ylim=None):
    x = np.arange(len(conditions))
    b1 = ax.bar(x - BAR_WIDTH/2, sac_vals, BAR_WIDTH,
                label="SAC", color=SAC_COLOR, alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + BAR_WIDTH/2, td3_vals, BAR_WIDTH,
                label="TD3", color=TD3_COLOR, alpha=0.88, edgecolor="white")

    # Value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(ylim or (0, None))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Plot 1-3: grouped bar charts
# ---------------------------------------------------------------------------
def plot_bars(df: pd.DataFrame, out_dir: str, show: bool):
    conditions = sorted(df["condition"].unique())
    labels     = [CONDITION_LABELS.get(c, c) for c in conditions]

    def get(algo, metric):
        return [float(df[(df["algo"]==algo) & (df["condition"]==c)][metric].values[0])
                if len(df[(df["algo"]==algo) & (df["condition"]==c)]) > 0 else 0.0
                for c in conditions]

    metrics = [
        ("success_rate",   "Success Rate",   (0, 1.05), "success_rate"),
        ("collision_rate", "Collision Rate", (0, 1.05), "collision_rate"),
        ("avg_reward",     "Avg Reward",     None,      "avg_reward"),
    ]

    for metric, ylabel, ylim, fname in metrics:
        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        _bar_group(ax, labels,
                   get("SAC", metric), get("TD3", metric),
                   ylabel, f"SAC vs TD3 — {ylabel} (Unseen Conditions)", ylim)
        plt.tight_layout()
        path = os.path.join(out_dir, f"sac_td3_{fname}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        if show:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Plot 4: radar chart
# ---------------------------------------------------------------------------
def plot_radar(df: pd.DataFrame, out_dir: str, show: bool):
    conditions = sorted(df["condition"].unique())
    n = len(conditions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    def get_success(algo):
        vals = [float(df[(df["algo"]==algo) & (df["condition"]==c)]["success_rate"].values[0])
                if len(df[(df["algo"]==algo) & (df["condition"]==c)]) > 0 else 0.0
                for c in conditions]
        return vals + vals[:1]

    fig, ax = plt.subplots(figsize=FIGSIZE_RADAR,
                            subplot_kw=dict(polar=True))

    sac_vals = get_success("SAC")
    td3_vals = get_success("TD3")

    ax.plot(angles, sac_vals, "o-", color=SAC_COLOR, linewidth=2, label="SAC")
    ax.fill(angles, sac_vals, color=SAC_COLOR, alpha=0.20)
    ax.plot(angles, td3_vals, "s-", color=TD3_COLOR, linewidth=2, label="TD3")
    ax.fill(angles, td3_vals, color=TD3_COLOR, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
    ax.set_title("Success Rate — Generalization Profile",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "sac_td3_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5: aggregate summary
# ---------------------------------------------------------------------------
def plot_aggregate(df: pd.DataFrame, out_dir: str, show: bool):
    agg = df.groupby("algo")[["success_rate","collision_rate"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_AGG)

    for ax, metric, ylabel, ylim in [
        (axes[0], "success_rate",   "Mean Success Rate",   (0, 1.0)),
        (axes[1], "collision_rate", "Mean Collision Rate", (0, 1.0)),
    ]:
        colors = [SAC_COLOR if a == "SAC" else TD3_COLOR
                  for a in agg["algo"]]
        bars = ax.bar(agg["algo"], agg[metric], color=colors,
                      alpha=0.88, edgecolor="white", width=0.4)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(ylim)
        ax.set_title(f"Aggregate {ylabel}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("SAC vs TD3 — Mean Across All Unseen Conditions",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "sac_td3_aggregate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 6: combined 2x2 summary panel
# ---------------------------------------------------------------------------
def plot_panel(df: pd.DataFrame, out_dir: str, show: bool):
    conditions = sorted(df["condition"].unique())
    labels     = [CONDITION_LABELS.get(c, c) for c in conditions]

    def get(algo, metric):
        return [float(df[(df["algo"]==algo) & (df["condition"]==c)][metric].values[0])
                if len(df[(df["algo"]==algo) & (df["condition"]==c)]) > 0 else 0.0
                for c in conditions]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("SAC vs TD3 — Unseen Generalization Evaluation",
                 fontsize=14, fontweight="bold")

    _bar_group(axes[0,0], labels, get("SAC","success_rate"),
               get("TD3","success_rate"), "Success Rate",
               "Success Rate per Condition", (0, 1.1))

    _bar_group(axes[0,1], labels, get("SAC","collision_rate"),
               get("TD3","collision_rate"), "Collision Rate",
               "Collision Rate per Condition", (0, 1.1))

    _bar_group(axes[1,0], labels, get("SAC","avg_reward"),
               get("TD3","avg_reward"), "Avg Reward",
               "Average Reward per Condition")

    _bar_group(axes[1,1], labels, get("SAC","avg_ep_length"),
               get("TD3","avg_ep_length"), "Avg Episode Length",
               "Episode Length per Condition")

    plt.tight_layout()
    path = os.path.join(out_dir, "sac_td3_panel.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV not found: {args.csv}\n"
            "Run eval_sac_vs_td3.py first to generate it."
        )

    df = load(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df[["algo","condition","success_rate","collision_rate","avg_reward"]].to_string(index=False))

    plot_bars(df, args.out_dir, args.show)
    plot_radar(df, args.out_dir, args.show)
    plot_aggregate(df, args.out_dir, args.show)
    plot_panel(df, args.out_dir, args.show)

    print(f"\nAll plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
