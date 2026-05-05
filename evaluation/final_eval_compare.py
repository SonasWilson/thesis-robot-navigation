import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from v5 import SafetySphereNavEnv


def make_env(render_mode=None):
    return SafetySphereNavEnv(render_mode=render_mode)


def load_model(algo: str, model_path: str, env):
    algo = algo.lower()
    if algo == "td3":
        return TD3.load(model_path, env=env, device="auto")
    if algo == "sac":
        return SAC.load(model_path, env=env, device="auto")
    raise ValueError(f"Unsupported algo '{algo}'. Use 'td3' or 'sac'.")


def evaluate_single_algo(
    algo: str,
    model_path: str,
    vecnorm_path: str,
    n_eval_episodes: int,
    render: bool = False,
) -> Dict:
    render_mode = "human" if render else None
    eval_env = DummyVecEnv([lambda: make_env(render_mode=render_mode)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = load_model(algo, model_path, eval_env)

    episode_rows: List[Dict] = []
    successes = 0
    collisions = 0
    timeouts = 0

    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done = [False]
        episode_return = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_return += float(reward[0])

        episode_info = info[0]
        success = float(episode_info.get("success", 0.0))
        collision = float(episode_info.get("collision", 0.0))
        path_length = float(episode_info.get("path_length", 0.0))
        episode_len = int(episode_info.get("episode_length", 0))
        final_dist = float(episode_info.get("dist_to_goal", 0.0))

        successes += int(success > 0.5)
        collisions += int(collision > 0.5)
        if success < 0.5 and collision < 0.5:
            timeouts += 1

        episode_rows.append(
            {
                "algo": algo.upper(),
                "episode": ep + 1,
                "episode_reward": episode_return,
                "episode_length_steps": episode_len,
                "path_length": path_length,
                "final_dist_to_goal": final_dist,
                "success": float(success > 0.5),
                "collision": float(collision > 0.5),
                "timeout": float((success < 0.5) and (collision < 0.5)),
            }
        )

    eval_env.close()

    ep_df = pd.DataFrame(episode_rows)
    metrics = {
        "algo": algo.upper(),
        "n_eval_episodes": int(n_eval_episodes),
        "success_rate": float(ep_df["success"].mean()),
        "collision_rate": float(ep_df["collision"].mean()),
        "timeout_rate": float(ep_df["timeout"].mean()),
        "avg_episode_reward": float(ep_df["episode_reward"].mean()),
        "std_episode_reward": float(ep_df["episode_reward"].std(ddof=0)),
        "avg_episode_length_steps": float(ep_df["episode_length_steps"].mean()),
        "avg_path_length": float(ep_df["path_length"].mean()),
        "avg_final_dist_to_goal": float(ep_df["final_dist_to_goal"].mean()),
    }
    return {"metrics": metrics, "episodes_df": ep_df}


def _bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_comparison(summary_df: pd.DataFrame, episodes_df: pd.DataFrame, out_dir: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: Key thesis metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "TD3 vs SAC: Final Evaluation Comparison", fontsize=16, fontweight="bold"
    )

    x = np.arange(len(summary_df))
    algos = summary_df["algo"].tolist()
    colors = ["#1f77b4", "#2ca02c"]

    bars1 = axes[0, 0].bar(
        x, summary_df["success_rate"], color=colors[: len(x)], width=0.6
    )
    axes[0, 0].set_title("Success Rate", fontweight="bold")
    axes[0, 0].set_xticks(x, algos)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Rate")
    _bar_labels(axes[0, 0], bars1, fmt="{:.3f}")

    bars2 = axes[0, 1].bar(
        x, summary_df["collision_rate"], color=colors[: len(x)], width=0.6
    )
    axes[0, 1].set_title("Collision Rate", fontweight="bold")
    axes[0, 1].set_xticks(x, algos)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Rate")
    _bar_labels(axes[0, 1], bars2, fmt="{:.3f}")

    bars3 = axes[1, 0].bar(
        x, summary_df["avg_path_length"], color=colors[: len(x)], width=0.6
    )
    axes[1, 0].set_title("Average Path Length", fontweight="bold")
    axes[1, 0].set_xticks(x, algos)
    axes[1, 0].set_ylabel("Distance")
    _bar_labels(axes[1, 0], bars3, fmt="{:.2f}")

    bars4 = axes[1, 1].bar(
        x,
        summary_df["avg_episode_reward"],
        yerr=summary_df["std_episode_reward"],
        capsize=6,
        color=colors[: len(x)],
        width=0.6,
    )
    axes[1, 1].set_title("Average Episode Reward (+/- std)", fontweight="bold")
    axes[1, 1].set_xticks(x, algos)
    axes[1, 1].set_ylabel("Reward")
    _bar_labels(axes[1, 1], bars4, fmt="{:.2f}")

    fig.savefig(
        os.path.join(out_dir, "thesis_metrics_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(out_dir, "thesis_metrics_comparison.pdf"), bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 2: Episode reward distributions
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    grouped = [
        episodes_df.loc[episodes_df["algo"] == algo, "episode_reward"].values
        for algo in algos
    ]
    bp = ax2.boxplot(grouped, labels=algos, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], colors[: len(grouped)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_title("Episode Reward Distribution (Final Evaluation)", fontweight="bold")
    ax2.set_ylabel("Episode Reward")
    ax2.grid(alpha=0.3)
    fig2.savefig(
        os.path.join(out_dir, "episode_reward_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig2.savefig(
        os.path.join(out_dir, "episode_reward_distribution.pdf"), bbox_inches="tight"
    )
    plt.close(fig2)

    # Figure 3: Cumulative success trend
    fig3, ax3 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for color, algo in zip(colors[: len(algos)], algos):
        algo_df = episodes_df[episodes_df["algo"] == algo].sort_values("episode")
        cumulative_success = algo_df["success"].cumsum() / np.arange(
            1, len(algo_df) + 1
        )
        ax3.plot(
            algo_df["episode"],
            cumulative_success,
            label=algo,
            linewidth=2.5,
            color=color,
        )
    ax3.set_title(
        "Cumulative Success Rate Across Evaluation Episodes", fontweight="bold"
    )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cumulative Success Rate")
    ax3.set_ylim(0, 1.0)
    ax3.legend(frameon=True)
    ax3.grid(alpha=0.3)
    fig3.savefig(
        os.path.join(out_dir, "cumulative_success_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig3.savefig(
        os.path.join(out_dir, "cumulative_success_curve.pdf"), bbox_inches="tight"
    )
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare TD3 vs SAC for thesis-quality reporting"
    )
    parser.add_argument("--td3_model_path", type=str, default="td3_navigation.zip")
    parser.add_argument("--td3_vecnorm_path", type=str, default="vecnormalize_td3.pkl")
    parser.add_argument("--sac_model_path", type=str, default="sac_navigation.zip")
    parser.add_argument("--sac_vecnorm_path", type=str, default="vecnormalize_sac.pkl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./final_eval_compare")
    args = parser.parse_args()

    required_files = [
        args.td3_model_path,
        args.td3_vecnorm_path,
        args.sac_model_path,
        args.sac_vecnorm_path,
    ]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    td3_result = evaluate_single_algo(
        "td3",
        model_path=args.td3_model_path,
        vecnorm_path=args.td3_vecnorm_path,
        n_eval_episodes=args.episodes,
        render=args.render,
    )
    sac_result = evaluate_single_algo(
        "sac",
        model_path=args.sac_model_path,
        vecnorm_path=args.sac_vecnorm_path,
        n_eval_episodes=args.episodes,
        render=args.render,
    )

    summary_df = pd.DataFrame([td3_result["metrics"], sac_result["metrics"]])
    episodes_df = pd.concat(
        [td3_result["episodes_df"], sac_result["episodes_df"]], ignore_index=True
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "comparison_summary.csv")
    episodes_path = os.path.join(args.out_dir, "comparison_episode_metrics.csv")
    json_path = os.path.join(args.out_dir, "comparison_summary.json")

    summary_df.to_csv(summary_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    plot_comparison(summary_df, episodes_df, args.out_dir)

    print("=== TD3 vs SAC Final Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved episode CSV: {episodes_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved plots (PNG + PDF) to: {args.out_dir}")


if __name__ == "__main__":
    main()
