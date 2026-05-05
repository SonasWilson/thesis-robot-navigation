"""
Publication-quality figures for the generalization assessment.

Reads CSVs from ./generalization_results/ and produces:
  Figure 4 — Zero-Shot Radar Plot
  Figure 5 — Arena Scale Cliff
  Figure 6 — Multi-Axis Stress Sweeps (5-panel)
  Figure 7 — Adaptation Efficiency (warm-up bars)
  Figure 8 — Generalization Boundary Heatmap
  Figure 9 — Episode Reward vs Path Length scatter

Usage:
    python plot_generalization_figures.py --out_dir ./figures --show
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":     150,
})

RESULT_DIR = "./generalization_results"
TRAINING_ARENA = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}\n"
                                "Run generalization_assessment.py first.")
    return pd.read_csv(path)


def save(fig, out_dir: str, name: str, show: bool):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Zero-Shot Radar Plot
# ---------------------------------------------------------------------------

def fig4_radar(df_zs: pd.DataFrame, out_dir: str, show: bool):
    """Radar chart: one axis per OOD dimension, success rate + 95% CI."""

    # Select representative config per axis
    axes_data = [
        ("Density\n(ZS-D3)",  "ZS-D3"),
        ("Speed\n(ZS-S2)",    "ZS-S2"),
        ("Arena-small\n(ZS-A1)", "ZS-A1"),
        ("Arena-large\n(ZS-A2)", "ZS-A2"),
        ("Motion\n(ZS-M1)",   "ZS-M1"),
        ("Combined\n(ZS-H2)", "ZS-H2"),
    ]

    labels  = [a[0] for a in axes_data]
    ids     = [a[1] for a in axes_data]
    n_axes  = len(labels)

    vals, lo, hi = [], [], []
    for cid in ids:
        row = df_zs[df_zs["config_id"] == cid].iloc[0]
        vals.append(float(row["success_rate"]))
        lo.append(float(row["success_rate_ci95_lo"]))
        hi.append(float(row["success_rate_ci95_hi"]))

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    vals_c = vals + vals[:1]
    lo_c   = lo   + lo[:1]
    hi_c   = hi   + hi[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Fill CI band
    ax.fill_between(angles, lo_c, hi_c, color="#2196F3", alpha=0.15, label="95% CI")
    # Main line
    ax.plot(angles, vals_c, "o-", color="#2196F3", linewidth=2.5, markersize=7,
            label="Success Rate", zorder=3)
    # Annotate each point
    for ang, val, cid in zip(angles[:-1], vals, ids):
        ax.annotate(f"{val:.1%}",
                    xy=(ang, val),
                    xytext=(ang, val + 0.06),
                    ha="center", va="center",
                    fontsize=9, color="#1565C0", fontweight="bold")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_title("Figure 4: Zero-Shot Generalization Profile",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(alpha=0.3)

    save(fig, out_dir, "fig4_radar_zero_shot", show)


# ---------------------------------------------------------------------------
# Figure 5 — Arena Scale Cliff
# ---------------------------------------------------------------------------

def fig5_arena_cliff(df_st: pd.DataFrame, out_dir: str, show: bool):
    """Line plot: success rate vs arena size with training boundary marked."""

    arena_df = df_st[df_st["config_id"].str.startswith("ST-SZ-")].copy()
    arena_df = arena_df.sort_values("arena_size")

    x   = arena_df["arena_size"].values
    y   = arena_df["success_rate"].values
    ylo = arena_df["success_rate_ci95_lo"].values
    yhi = arena_df["success_rate_ci95_hi"].values

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(x, ylo, yhi, color="#E53935", alpha=0.15, label="95% CI")
    ax.plot(x, y, "o-", color="#E53935", linewidth=2.5, markersize=9,
            zorder=3, label="Success Rate")

    # Training arena vertical line
    ax.axvline(TRAINING_ARENA, color="#43A047", linewidth=2, linestyle="--",
               label=f"Training arena ({int(TRAINING_ARENA)}×{int(TRAINING_ARENA)})")
    ax.text(TRAINING_ARENA + 1.5, 0.05, "Training\narena",
            color="#43A047", fontsize=9, va="bottom")

    # Annotate cliff values
    cliff_annotations = {
        10.0:  ("18%", "below"),
        25.0:  ("70%", "above"),
        50.0:  ("98%", "above"),
        70.0:  ("86%", "above"),
        100.0: ("36%", "below"),
    }
    for sz, (label, pos) in cliff_annotations.items():
        row = arena_df[arena_df["arena_size"] == sz]
        if row.empty:
            continue
        yv = float(row["success_rate"].values[0])
        offset = 0.05 if pos == "above" else -0.07
        ax.annotate(label, xy=(sz, yv), xytext=(sz, yv + offset),
                    ha="center", fontsize=10, fontweight="bold",
                    color="#B71C1C",
                    arrowprops=dict(arrowstyle="-", color="#B71C1C", lw=1))

    # Shade failure zones
    ax.axvspan(0, 25, alpha=0.06, color="red")
    ax.axvspan(70, 110, alpha=0.06, color="red")
    ax.text(17, 0.92, "Failure\nzone", ha="center", fontsize=8,
            color="#B71C1C", alpha=0.7)
    ax.text(90, 0.92, "Failure\nzone", ha="center", fontsize=8,
            color="#B71C1C", alpha=0.7)

    ax.set_xlabel("Arena Size (m × m)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(5, 108)
    ax.set_xticks([10, 25, 50, 70, 100])
    ax.set_xticklabels(["10×10", "25×25", "50×50\n(train)", "70×70", "100×100"])
    ax.set_title("Figure 5: Arena Scale Cliff — Performance vs Arena Size",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, out_dir, "fig5_arena_cliff", show)


# ---------------------------------------------------------------------------
# Figure 6 — Multi-Axis Stress Sweeps (5-panel)
# ---------------------------------------------------------------------------

def fig6_stress_sweeps(df_st: pd.DataFrame, out_dir: str, show: bool):
    """5-panel subplot: one per stress axis."""

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Figure 6: Systematic OOD Stress Sweeps",
                 fontsize=14, fontweight="bold")

    panels = [
        ("ST-DYN-",  "num_dynamic_obstacles", "Dynamic Obstacles",
         "# Dynamic Agents", "#2196F3"),
        ("ST-SPD-",  "dynamic_speed",          "Dynamic Speed",
         "Speed Scale", "#E53935"),
        ("ST-STA-",  "num_static_obstacles",   "Static Obstacles",
         "# Static Obstacles", "#FF9800"),
        ("ST-SZ-",   "arena_size",             "Arena Size",
         "Arena Size (m)", "#9C27B0"),
        ("ST-ORB-",  "orbital_fraction",       "Orbital Fraction",
         "Orbital Fraction", "#009688"),
    ]

    for ax, (prefix, xcol, title, xlabel, color) in zip(axes, panels):
        sub = df_st[df_st["config_id"].str.startswith(prefix)].copy()
        sub = sub.sort_values(xcol)

        x   = sub[xcol].values
        y   = sub["success_rate"].values
        ylo = sub["success_rate_ci95_lo"].values
        yhi = sub["success_rate_ci95_hi"].values

        ax.fill_between(x, ylo, yhi, color=color, alpha=0.15)
        ax.plot(x, y, "o-", color=color, linewidth=2.2, markersize=7)

        # Mark training value
        if xcol == "arena_size":
            ax.axvline(50.0, color="green", linewidth=1.5,
                       linestyle="--", alpha=0.7, label="Train")
            ax.legend(fontsize=8)
        elif xcol == "dynamic_speed":
            ax.axvline(1.20, color="green", linewidth=1.5,
                       linestyle="--", alpha=0.7)
        elif xcol == "num_dynamic_obstacles":
            ax.axvline(4, color="green", linewidth=1.5,
                       linestyle="--", alpha=0.7)
        elif xcol == "num_static_obstacles":
            ax.axvline(3, color="green", linewidth=1.5,
                       linestyle="--", alpha=0.7)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Success Rate" if ax == axes[0] else "")
        ax.set_ylim(0, 1.08)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate min/max
        ax.annotate(f"{y.min():.0%}", xy=(x[y.argmin()], y.min()),
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold")
        ax.annotate(f"{y.max():.0%}", xy=(x[y.argmax()], y.max()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    save(fig, out_dir, "fig6_stress_sweeps", show)


# ---------------------------------------------------------------------------
# Figure 7 — Adaptation Efficiency (warm-up bars)
# ---------------------------------------------------------------------------

def fig7_warmup_bars(df_wu: pd.DataFrame, out_dir: str, show: bool):
    """Grouped bar chart: zero-shot vs post-warmup per config."""

    configs = df_wu["config_id"].unique()
    # Separate before/after
    before = df_wu[df_wu["mode"] == "warm_up_zero_shot"].set_index("config_id")
    after  = df_wu[df_wu["mode"] == "warm_up_after"].set_index("config_id")

    common = [c for c in configs if c in before.index and c in after.index]
    labels = [before.loc[c, "description"] if isinstance(
        before.loc[c, "description"], str)
        else before.loc[c, "description"].iloc[0] for c in common]

    x = np.arange(len(common))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    b_vals  = [float(before.loc[c, "success_rate"])
               if isinstance(before.loc[c, "success_rate"], float)
               else float(before.loc[c, "success_rate"].iloc[0]) for c in common]
    b_lo    = [float(before.loc[c, "success_rate_ci95_lo"])
               if isinstance(before.loc[c, "success_rate_ci95_lo"], float)
               else float(before.loc[c, "success_rate_ci95_lo"].iloc[0]) for c in common]
    b_hi    = [float(before.loc[c, "success_rate_ci95_hi"])
               if isinstance(before.loc[c, "success_rate_ci95_hi"], float)
               else float(before.loc[c, "success_rate_ci95_hi"].iloc[0]) for c in common]

    a_vals  = [float(after.loc[c, "success_rate"])
               if isinstance(after.loc[c, "success_rate"], float)
               else float(after.loc[c, "success_rate"].iloc[0]) for c in common]
    a_lo    = [float(after.loc[c, "success_rate_ci95_lo"])
               if isinstance(after.loc[c, "success_rate_ci95_lo"], float)
               else float(after.loc[c, "success_rate_ci95_lo"].iloc[0]) for c in common]
    a_hi    = [float(after.loc[c, "success_rate_ci95_hi"])
               if isinstance(after.loc[c, "success_rate_ci95_hi"], float)
               else float(after.loc[c, "success_rate_ci95_hi"].iloc[0]) for c in common]

    b_err = [[v - l for v, l in zip(b_vals, b_lo)],
             [h - v for v, h in zip(b_vals, b_hi)]]
    a_err = [[v - l for v, l in zip(a_vals, a_lo)],
             [h - v for v, h in zip(a_vals, a_hi)]]

    bars1 = ax.bar(x - w/2, b_vals, w, label="Zero-Shot",
                   color="#2196F3", alpha=0.85, yerr=b_err,
                   capsize=5, error_kw=dict(elinewidth=1.5))
    bars2 = ax.bar(x + w/2, a_vals, w, label="Post Warm-Up (10k steps)",
                   color="#FF9800", alpha=0.85, yerr=a_err,
                   capsize=5, error_kw=dict(elinewidth=1.5))

    # Annotate delta
    for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
        delta = av - bv
        color = "#43A047" if delta >= 0 else "#E53935"
        ax.annotate(f"{delta:+.1%}",
                    xy=(x[i] + w/2, av + 0.03),
                    ha="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Figure 7: Adaptation Efficiency — Zero-Shot vs Post Warm-Up",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, out_dir, "fig7_warmup_bars", show)


# ---------------------------------------------------------------------------
# Figure 8 — Generalization Boundary Heatmap
# ---------------------------------------------------------------------------

def fig8_boundary_heatmap(df_zs: pd.DataFrame, out_dir: str, show: bool):
    """Heatmap: rows=OOD axes, columns=performance regime."""

    # Aggregate by axis
    axis_map = {
        "Density":    ["ZS-D1", "ZS-D2", "ZS-D3"],
        "Speed":      ["ZS-S1", "ZS-S2"],
        "Arena":      ["ZS-A1", "ZS-A2"],
        "Motion":     ["ZS-M1", "ZS-M2"],
        "Combined":   ["ZS-H1", "ZS-H2"],
    }
    regimes = ["Robust\n(>90%)", "Moderate\n(70–90%)", "Fail\n(<70%)"]

    # Build matrix: fraction of configs in each regime per axis
    matrix = []
    row_labels = []
    mean_vals = []
    for axis, cids in axis_map.items():
        sub = df_zs[df_zs["config_id"].isin(cids)]["success_rate"].values
        robust   = float(np.mean(sub >= 0.90))
        moderate = float(np.mean((sub >= 0.70) & (sub < 0.90)))
        fail     = float(np.mean(sub < 0.70))
        matrix.append([robust, moderate, fail])
        row_labels.append(axis)
        mean_vals.append(float(np.mean(sub)))

    mat = np.array(matrix)

    # Custom colormap: green→yellow→red
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ry", ["#E53935", "#FDD835", "#43A047"], N=256)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(regimes)):
            val = mat[i, j]
            text_color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    # Mean success rate on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels([f"μ={v:.1%}" for v in mean_vals], fontsize=10)

    plt.colorbar(im, ax=ax, label="Fraction of configs in regime",
                 shrink=0.8, pad=0.12)
    ax.set_title("Figure 8: Generalization Boundary Summary",
                 fontsize=13, fontweight="bold")

    save(fig, out_dir, "fig8_boundary_heatmap", show)


# ---------------------------------------------------------------------------
# Figure 9 — Episode Reward vs Path Length scatter
# ---------------------------------------------------------------------------

def fig9_reward_path_scatter(df_zs: pd.DataFrame, out_dir: str, show: bool):
    """Scatter: avg_reward vs avg_path_length, colored by difficulty."""

    # Difficulty proxy: collision_rate (higher = harder)
    x = df_zs["avg_episode_length"].values
    y = df_zs["avg_episode_reward"].values
    c = df_zs["collision_rate"].values
    labels = df_zs["config_id"].values

    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(x, y, c=c, cmap="RdYlGn_r", s=120,
                    vmin=0, vmax=0.5, edgecolors="white",
                    linewidths=0.8, zorder=3)

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, p(xline), "--", color="#555", linewidth=1.5,
            alpha=0.6, label=f"Trend (r={np.corrcoef(x, y)[0,1]:.2f})")

    # Label each point
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, alpha=0.8)

    plt.colorbar(sc, ax=ax, label="Collision Rate (difficulty proxy)")
    ax.set_xlabel("Average Episode Length (steps)", fontsize=12)
    ax.set_ylabel("Average Episode Reward", fontsize=12)
    ax.set_title("Figure 9: Navigation Efficiency Preserved Under OOD Conditions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, out_dir, "fig9_reward_path_scatter", show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, default="./generalization_results")
    p.add_argument("--out_dir",    type=str, default="./figures")
    p.add_argument("--show",       action="store_true")
    p.add_argument("--figs",       type=str, default="all",
                   help="Comma-separated list: 4,5,6,7,8,9 or 'all'")
    return p.parse_args()


def main():
    args = parse_args()
    global RESULT_DIR
    RESULT_DIR = args.result_dir

    figs = set(args.figs.split(",")) if args.figs != "all" else \
           {"4", "5", "6", "7", "8", "9"}

    print(f"Loading results from: {RESULT_DIR}")
    df_zs = load_csv("generalization_zero_shot.csv")
    df_st = load_csv("generalization_ood_stress.csv")
    df_wu = load_csv("generalization_warm_up.csv")

    if "4" in figs:
        print("\n--- Figure 4: Radar Plot ---")
        fig4_radar(df_zs, args.out_dir, args.show)

    if "5" in figs:
        print("\n--- Figure 5: Arena Cliff ---")
        fig5_arena_cliff(df_st, args.out_dir, args.show)

    if "6" in figs:
        print("\n--- Figure 6: Stress Sweeps ---")
        fig6_stress_sweeps(df_st, args.out_dir, args.show)

    if "7" in figs:
        print("\n--- Figure 7: Warm-Up Bars ---")
        fig7_warmup_bars(df_wu, args.out_dir, args.show)

    if "8" in figs:
        print("\n--- Figure 8: Boundary Heatmap ---")
        fig8_boundary_heatmap(df_zs, args.out_dir, args.show)

    if "9" in figs:
        print("\n--- Figure 9: Reward vs Path Scatter ---")
        fig9_reward_path_scatter(df_zs, args.out_dir, args.show)

    print(f"\nAll figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
