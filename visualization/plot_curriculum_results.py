import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot curriculum evaluation summary metrics by stage")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./models/curriculum_eval_summary.csv",
        help="CSV produced by curriculum_eval.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./models",
        help="Directory to save curriculum plots",
    )
    return parser.parse_args()


def plot_curriculum_metrics(df: pd.DataFrame, out_dir: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    os.makedirs(out_dir, exist_ok=True)

    df = df.sort_values("stage_id").reset_index(drop=True)
    x = df["stage_id"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Curriculum Progression (SAC)", fontsize=16, fontweight="bold")

    axes[0, 0].plot(x, df["success_rate"], marker="o", linewidth=2.3, color="tab:green")
    axes[0, 0].set_title("Success Rate by Stage", fontweight="bold")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_xticks(x)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, df["collision_rate"], marker="o", linewidth=2.3, color="tab:red")
    axes[0, 1].set_title("Collision Rate by Stage", fontweight="bold")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_xticks(x)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, df["avg_reward"], marker="o", linewidth=2.3, color="tab:purple")
    if "std_reward" in df.columns:
        axes[1, 0].fill_between(
            x,
            df["avg_reward"] - df["std_reward"],
            df["avg_reward"] + df["std_reward"],
            color="tab:purple",
            alpha=0.15,
            label="±1 std",
        )
        axes[1, 0].legend(frameon=True)
    axes[1, 0].set_title("Average Reward by Stage", fontweight="bold")
    axes[1, 0].set_xlabel("Stage")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].set_xticks(x)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(
        x, df["avg_episode_length"], marker="o", linewidth=2.3, color="tab:blue"
    )
    axes[1, 1].set_title("Average Episode Length by Stage", fontweight="bold")
    axes[1, 1].set_xlabel("Stage")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].set_xticks(x)
    axes[1, 1].grid(alpha=0.3)

    metrics_png = os.path.join(out_dir, "curriculum_stage_metrics.png")
    metrics_pdf = os.path.join(out_dir, "curriculum_stage_metrics.pdf")
    fig.savefig(metrics_png, dpi=300, bbox_inches="tight")
    fig.savefig(metrics_pdf, bbox_inches="tight")
    plt.close(fig)

    # Difficulty-aware trend chart.
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    difficulty = (
        df["num_static_obstacles"]
        + df["num_dynamic_obstacles"]
        + (df["arena_size"] / 10.0)
        + df["dynamic_speed"]
    )
    ax2.plot(difficulty, df["success_rate"], marker="o", linewidth=2.2, color="tab:green")
    for _, row in df.iterrows():
        ax2.annotate(
            f"S{int(row['stage_id'])}",
            (difficulty.loc[row.name], row["success_rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax2.set_title("Success Rate vs Curriculum Difficulty", fontweight="bold")
    ax2.set_xlabel("Difficulty Index (obstacles + arena scale + dynamic speed)")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3)

    diff_png = os.path.join(out_dir, "curriculum_difficulty_success_curve.png")
    diff_pdf = os.path.join(out_dir, "curriculum_difficulty_success_curve.pdf")
    fig2.savefig(diff_png, dpi=300, bbox_inches="tight")
    fig2.savefig(diff_pdf, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved plot: {metrics_png}")
    print(f"Saved plot: {metrics_pdf}")
    print(f"Saved plot: {diff_png}")
    print(f"Saved plot: {diff_pdf}")


def main():
    args = parse_args()
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required_cols = {
        "stage_id",
        "success_rate",
        "collision_rate",
        "avg_episode_length",
        "avg_reward",
        "num_static_obstacles",
        "num_dynamic_obstacles",
        "arena_size",
        "dynamic_speed",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in input CSV: {sorted(missing)}")

    plot_curriculum_metrics(df=df, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
