"""
Generalization study plot — compares zero-shot, random warmup,
and policy rollout warmup across V1, V2, V3.

Usage:
    python plot_generalization_study.py \
        --zeroshot    results_zeroshot.csv \
        --random      results_random_warmup.csv \
        --full        results_full_warmup.csv \
        --out_dir     ./plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = {
    "Zero-Shot":          "#E53935",   # red
    "Random Warmup":      "#FB8C00",   # orange
    "Policy Warmup":      "#43A047",   # green
}
ENVS = ["V1", "V2", "V3"]
ENV_LABELS = {
    "V1": "V1\nTopology Shift",
    "V2": "V2\nDynamics Shift",
    "V3": "V3\nCombined Stress",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zeroshot", type=str, default="results_zeroshot.csv")
    p.add_argument("--random",   type=str, default="results_random_warmup.csv")
    p.add_argument("--full",     type=str, default="results_full_warmup.csv")
    p.add_argument("--out_dir",  type=str, default="./plots")
    p.add_argument("--show",     action="store_true")
    return p.parse_args()


def load_all(args):
    dfs = {}
    for label, path in [
        ("Zero-Shot",     args.zeroshot),
        ("Random Warmup", args.random),
        ("Policy Warmup", args.full),
    ]:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping {label}")
            continue
        df = pd.read_csv(path)
        df["env"] = df["env"].str.upper().str.strip()
        dfs[label] = df
    return dfs


def get_val(df, env, metric):
    row = df[df["env"] == env]
    if row.empty:
        return 0.0
    return float(row[metric].values[0])


def plot_metric(dfs, metric, ylabel, title, ylim, out_path, show):
    envs   = [e for e in ENVS]
    labels = [ENV_LABELS[e] for e in envs]
    n_cond = len(dfs)
    n_env  = len(envs)
    total_w = 0.7
    w = total_w / n_cond
    offsets = np.linspace(-(total_w - w) / 2, (total_w - w) / 2, n_cond)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_env)

    for (cond_label, df), offset, color in zip(dfs.items(), offsets, CONDITIONS.values()):
        vals = [get_val(df, e, metric) for e in envs]
        bars = ax.bar(x + offset, vals, w, label=cond_label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_improvement(dfs, out_path, show):
    """Bar chart showing absolute improvement from zero-shot to policy warmup."""
    if "Zero-Shot" not in dfs or "Policy Warmup" not in dfs:
        print("Need both zero-shot and policy warmup CSVs for improvement plot.")
        return

    envs   = [e for e in ENVS]
    labels = [ENV_LABELS[e] for e in envs]
    zs = dfs["Zero-Shot"]
    pw = dfs["Policy Warmup"]

    improvements = [
        get_val(pw, e, "success_rate") - get_val(zs, e, "success_rate")
        for e in envs
    ]
    colors = ["#43A047" if v >= 0 else "#E53935" for v in improvements]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, improvements, color=colors, alpha=0.85,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.005 if val >= 0 else -0.02),
                f"{val:+.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Success Rate Improvement\n(Policy Warmup − Zero-Shot)", fontsize=10)
    ax.set_title("Normalizer Adaptation Benefit per Environment",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_table(dfs):
    print("\n=== Generalization Study Summary ===")
    print(f"{'Env':<6} {'Condition':<20} {'Success':>8} {'Collision':>10} {'Avg Reward':>12} {'Avg Len':>9}")
    print("-" * 70)
    for env in ENVS:
        for cond, df in dfs.items():
            s  = get_val(df, env, "success_rate")
            c  = get_val(df, env, "collision_rate")
            r  = get_val(df, env, "avg_reward")
            l  = get_val(df, env, "avg_ep_length")
            print(f"{env:<6} {cond:<20} {s:>8.3f} {c:>10.3f} {r:>12.1f} {l:>9.0f}")
        print()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dfs = load_all(args)

    if not dfs:
        print("No CSV files found. Run eval_unseen_v2.py first.")
        return

    print_table(dfs)

    plot_metric(dfs, "success_rate", "Success Rate",
                "Generalization Study — Success Rate by Condition",
                (0, 1.1),
                os.path.join(args.out_dir, "gen_study_success.png"), args.show)

    plot_metric(dfs, "collision_rate", "Collision Rate",
                "Generalization Study — Collision Rate by Condition",
                (0, 1.1),
                os.path.join(args.out_dir, "gen_study_collision.png"), args.show)

    plot_metric(dfs, "avg_reward", "Average Reward",
                "Generalization Study — Average Reward by Condition",
                None,
                os.path.join(args.out_dir, "gen_study_reward.png"), args.show)

    plot_improvement(dfs,
                     os.path.join(args.out_dir, "gen_study_improvement.png"),
                     args.show)

    print(f"\nAll plots saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
