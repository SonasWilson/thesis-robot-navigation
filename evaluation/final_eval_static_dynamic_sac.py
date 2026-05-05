import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dynamic_2 import DynamicObstacleNavEnv
from static_2 import StaticObstacleNavEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final SAC evaluation: static_2 vs dynamic_2 environments"
    )
    parser.add_argument("--seed", type=int, default=42, help="Model seed used in file naming")
    parser.add_argument("--episodes", type=int, default=200, help="Evaluation episodes per environment")
    parser.add_argument("--arena_size", type=float, default=10.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./final_eval_static_dynamic_sac")

    # Keep these aligned with defaults used in static_2_sac.py and dynamic_2_sac.py
    parser.add_argument(
        "--static_model_template",
        type=str,
        default="sac_static2_navigation_seed{seed}.zip",
    )
    parser.add_argument(
        "--static_vecnorm_template",
        type=str,
        default="vecnormalize_static2_sac_seed{seed}.pkl",
    )
    parser.add_argument(
        "--dynamic_model_template",
        type=str,
        default="sac_dynamic2_navigation_seed{seed}.zip",
    )
    parser.add_argument(
        "--dynamic_vecnorm_template",
        type=str,
        default="vecnormalize_dynamic2_sac_seed{seed}.pkl",
    )
    return parser.parse_args()


def build_path(template: str, seed: int) -> str:
    return template.format(seed=seed)


def _make_env(env_type: str, arena_size: float, render_mode=None):
    if env_type == "static2":
        return StaticObstacleNavEnv(render_mode=render_mode, arena_size=arena_size)
    if env_type == "dynamic2":
        return DynamicObstacleNavEnv(render_mode=render_mode, arena_size=arena_size)
    raise ValueError(f"Unsupported env type: {env_type}")


def evaluate_sac(
    env_type: str,
    model_path: str,
    vecnorm_path: str,
    n_eval_episodes: int,
    arena_size: float,
    render: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    render_mode = "human" if render else None
    env = DummyVecEnv(
        [lambda: _make_env(env_type=env_type, arena_size=arena_size, render_mode=render_mode)]
    )
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = SAC.load(model_path, env=env, device="auto")

    rows: List[Dict] = []
    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = [False]
        episode_return = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_return += float(reward[0])

        ep_info = info[0]
        success = float(ep_info.get("success", 0.0))
        collision = float(ep_info.get("collision", 0.0))
        timeout = float((success < 0.5) and (collision < 0.5))

        rows.append(
            {
                "env_type": env_type,
                "episode": ep + 1,
                "episode_reward": episode_return,
                "episode_length_steps": int(ep_info.get("episode_length", 0)),
                "path_length": float(ep_info.get("path_length", 0.0)),
                "final_dist_to_goal": float(ep_info.get("dist_to_goal", 0.0)),
                "success": float(success > 0.5),
                "collision": float(collision > 0.5),
                "timeout": timeout,
            }
        )

    env.close()

    ep_df = pd.DataFrame(rows)
    summary = {
        "env_type": env_type,
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
    return summary, ep_df


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
    colors = ["#1f77b4", "#2ca02c"]
    labels = ["Static2-SAC", "Dynamic2-SAC"]

    # Figure 1: Core comparison metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("SAC Final Evaluation: Static2 vs Dynamic2", fontsize=16, fontweight="bold")
    x = np.arange(len(summary_df))

    bars1 = axes[0, 0].bar(x, summary_df["success_rate"], color=colors, width=0.6)
    axes[0, 0].set_title("Success Rate", fontweight="bold")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, 1.05)
    _bar_labels(axes[0, 0], bars1)

    bars2 = axes[0, 1].bar(x, summary_df["collision_rate"], color=colors, width=0.6)
    axes[0, 1].set_title("Collision Rate", fontweight="bold")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0, 1.05)
    _bar_labels(axes[0, 1], bars2)

    bars3 = axes[1, 0].bar(x, summary_df["avg_path_length"], color=colors, width=0.6)
    axes[1, 0].set_title("Average Path Length", fontweight="bold")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Distance")
    _bar_labels(axes[1, 0], bars3, fmt="{:.2f}")

    bars4 = axes[1, 1].bar(
        x,
        summary_df["avg_episode_reward"],
        yerr=summary_df["std_episode_reward"],
        capsize=6,
        color=colors,
        width=0.6,
    )
    axes[1, 1].set_title("Average Episode Reward (+/- std)", fontweight="bold")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Reward")
    _bar_labels(axes[1, 1], bars4, fmt="{:.2f}")

    fig.savefig(os.path.join(out_dir, "sac_static_dynamic_metrics.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "sac_static_dynamic_metrics.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Episode reward distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    grouped = [
        episodes_df.loc[episodes_df["env_type"] == "static2", "episode_reward"].values,
        episodes_df.loc[episodes_df["env_type"] == "dynamic2", "episode_reward"].values,
    ]
    bp = ax2.boxplot(grouped, labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_title("Episode Reward Distribution", fontweight="bold")
    ax2.set_ylabel("Episode Reward")
    ax2.grid(alpha=0.3)
    fig2.savefig(os.path.join(out_dir, "sac_static_dynamic_reward_distribution.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "sac_static_dynamic_reward_distribution.pdf"), bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Cumulative success trend
    fig3, ax3 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for env_type, color, label in zip(["static2", "dynamic2"], colors, labels):
        env_df = episodes_df[episodes_df["env_type"] == env_type].sort_values("episode")
        cumulative_success = env_df["success"].cumsum() / np.arange(1, len(env_df) + 1)
        ax3.plot(
            env_df["episode"],
            cumulative_success,
            label=label,
            linewidth=2.5,
            color=color,
        )
    ax3.set_title("Cumulative Success Rate Over Evaluation Episodes", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cumulative Success Rate")
    ax3.set_ylim(0, 1.0)
    ax3.legend(frameon=True)
    ax3.grid(alpha=0.3)
    fig3.savefig(os.path.join(out_dir, "sac_static_dynamic_cumulative_success.png"), dpi=300, bbox_inches="tight")
    fig3.savefig(os.path.join(out_dir, "sac_static_dynamic_cumulative_success.pdf"), bbox_inches="tight")
    plt.close(fig3)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    static_model_path = build_path(args.static_model_template, args.seed)
    static_vecnorm_path = build_path(args.static_vecnorm_template, args.seed)
    dynamic_model_path = build_path(args.dynamic_model_template, args.seed)
    dynamic_vecnorm_path = build_path(args.dynamic_vecnorm_template, args.seed)

    required_files = [
        static_model_path,
        static_vecnorm_path,
        dynamic_model_path,
        dynamic_vecnorm_path,
    ]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    static_summary, static_episodes = evaluate_sac(
        env_type="static2",
        model_path=static_model_path,
        vecnorm_path=static_vecnorm_path,
        n_eval_episodes=args.episodes,
        arena_size=args.arena_size,
        render=args.render,
    )
    dynamic_summary, dynamic_episodes = evaluate_sac(
        env_type="dynamic2",
        model_path=dynamic_model_path,
        vecnorm_path=dynamic_vecnorm_path,
        n_eval_episodes=args.episodes,
        arena_size=args.arena_size,
        render=args.render,
    )

    summary_df = pd.DataFrame([static_summary, dynamic_summary])
    episodes_df = pd.concat([static_episodes, dynamic_episodes], ignore_index=True)

    summary_csv = os.path.join(args.out_dir, "sac_static_dynamic_summary.csv")
    episodes_csv = os.path.join(args.out_dir, "sac_static_dynamic_episode_metrics.csv")
    summary_json = os.path.join(args.out_dir, "sac_static_dynamic_summary.json")

    summary_df.to_csv(summary_csv, index=False)
    episodes_df.to_csv(episodes_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)
    plot_comparison(summary_df, episodes_df, args.out_dir)

    print("=== SAC Final Evaluation: Static2 vs Dynamic2 ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved episode CSV: {episodes_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved plots (PNG + PDF): {args.out_dir}")


if __name__ == "__main__":
    main()
