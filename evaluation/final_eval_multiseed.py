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


def evaluate_one_seed(
    algo: str,
    model_path: str,
    vecnorm_path: str,
    n_eval_episodes: int,
    render: bool = False,
) -> Dict:
    render_mode = "human" if render else None
    env = DummyVecEnv([lambda: make_env(render_mode=render_mode)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = load_model(algo, model_path, env)

    rewards, lengths, paths, dists = [], [], [], []
    successes = 0
    collisions = 0
    timeouts = 0

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = [False]
        ep_return = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward[0])

        ep_info = info[0]
        success = float(ep_info.get("success", 0.0))
        collision = float(ep_info.get("collision", 0.0))
        successes += int(success > 0.5)
        collisions += int(collision > 0.5)
        if success < 0.5 and collision < 0.5:
            timeouts += 1

        rewards.append(ep_return)
        lengths.append(float(ep_info.get("episode_length", 0.0)))
        paths.append(float(ep_info.get("path_length", 0.0)))
        dists.append(float(ep_info.get("dist_to_goal", 0.0)))

    env.close()

    return {
        "success_rate": float(successes / n_eval_episodes),
        "collision_rate": float(collisions / n_eval_episodes),
        "timeout_rate": float(timeouts / n_eval_episodes),
        "avg_path_length": float(np.mean(paths)),
        "avg_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "avg_episode_length_steps": float(np.mean(lengths)),
        "avg_final_dist_to_goal": float(np.mean(dists)),
    }


def aggregate_seed_metrics(seed_df: pd.DataFrame) -> Dict:
    keys = [
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "avg_path_length",
        "avg_episode_reward",
        "avg_episode_length_steps",
        "avg_final_dist_to_goal",
    ]
    out = {}
    for key in keys:
        out[f"{key}_mean"] = float(seed_df[key].mean())
        out[f"{key}_std"] = float(seed_df[key].std(ddof=0))
    return out


def fmt_pct(mean: float, std: float) -> str:
    return f"{100.0 * mean:.1f}% ± {100.0 * std:.1f}%"


def fmt_val(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def print_human_summary(algo: str, agg: Dict, n_episodes: int, seeds: List[int]):
    print(
        f"\n=== {algo.upper()} | {n_episodes} episodes x {len(seeds)} seeds ({seeds}) ==="
    )
    print(
        f"Success rate = {fmt_pct(agg['success_rate_mean'], agg['success_rate_std'])}"
    )
    print(
        f"Collision rate = {fmt_pct(agg['collision_rate_mean'], agg['collision_rate_std'])}"
    )
    print(
        f"Avg path length = {fmt_val(agg['avg_path_length_mean'], agg['avg_path_length_std'])}"
    )
    print(
        f"Avg reward = {fmt_val(agg['avg_episode_reward_mean'], agg['avg_episode_reward_std'])}"
    )
    print(
        f"Timeout rate = {fmt_pct(agg['timeout_rate_mean'], agg['timeout_rate_std'])}"
    )


def plot_multiseed(algo_to_agg: Dict[str, Dict], out_dir: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "Multi-Seed Final Evaluation (Mean ± Std Across Seeds)",
        fontsize=16,
        fontweight="bold",
    )

    algos = list(algo_to_agg.keys())
    x = np.arange(len(algos))
    colors = ["#1f77b4", "#2ca02c"]

    sr_mean = [algo_to_agg[a]["success_rate_mean"] for a in algos]
    sr_std = [algo_to_agg[a]["success_rate_std"] for a in algos]
    cr_mean = [algo_to_agg[a]["collision_rate_mean"] for a in algos]
    cr_std = [algo_to_agg[a]["collision_rate_std"] for a in algos]
    pl_mean = [algo_to_agg[a]["avg_path_length_mean"] for a in algos]
    pl_std = [algo_to_agg[a]["avg_path_length_std"] for a in algos]
    rw_mean = [algo_to_agg[a]["avg_episode_reward_mean"] for a in algos]
    rw_std = [algo_to_agg[a]["avg_episode_reward_std"] for a in algos]

    axes[0, 0].bar(
        x, sr_mean, yerr=sr_std, capsize=6, color=colors[: len(algos)], width=0.6
    )
    axes[0, 0].set_title("Success Rate", fontweight="bold")
    axes[0, 0].set_xticks(x, [a.upper() for a in algos])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Rate")

    axes[0, 1].bar(
        x, cr_mean, yerr=cr_std, capsize=6, color=colors[: len(algos)], width=0.6
    )
    axes[0, 1].set_title("Collision Rate", fontweight="bold")
    axes[0, 1].set_xticks(x, [a.upper() for a in algos])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Rate")

    axes[1, 0].bar(
        x, pl_mean, yerr=pl_std, capsize=6, color=colors[: len(algos)], width=0.6
    )
    axes[1, 0].set_title("Average Path Length", fontweight="bold")
    axes[1, 0].set_xticks(x, [a.upper() for a in algos])
    axes[1, 0].set_ylabel("Distance")

    axes[1, 1].bar(
        x, rw_mean, yerr=rw_std, capsize=6, color=colors[: len(algos)], width=0.6
    )
    axes[1, 1].set_title("Average Episode Reward", fontweight="bold")
    axes[1, 1].set_xticks(x, [a.upper() for a in algos])
    axes[1, 1].set_ylabel("Reward")

    png = os.path.join(out_dir, "multiseed_metrics_comparison.png")
    pdf = os.path.join(out_dir, "multiseed_metrics_comparison.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_path(template: str, seed: int) -> str:
    return template.format(seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed final evaluation for TD3 and SAC"
    )
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per seed")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
        help="Seeds to aggregate",
    )
    parser.add_argument(
        "--algos", type=str, nargs="+", default=["td3", "sac"], choices=["td3", "sac"]
    )
    parser.add_argument(
        "--td3_model_template", type=str, default="td3_navigation_seed{seed}.zip"
    )
    parser.add_argument(
        "--td3_vecnorm_template", type=str, default="vecnormalize_td3_seed{seed}.pkl"
    )
    parser.add_argument(
        "--sac_model_template", type=str, default="sac_navigation_seed{seed}.zip"
    )
    parser.add_argument(
        "--sac_vecnorm_template", type=str, default="vecnormalize_sac_seed{seed}.pkl"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./final_eval_multiseed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seed_rows = []
    algo_to_agg = {}

    for algo in args.algos:
        per_seed_metrics = []
        for seed in args.seeds:
            if algo == "td3":
                model_path = build_path(args.td3_model_template, seed)
                vecnorm_path = build_path(args.td3_vecnorm_template, seed)
            else:
                model_path = build_path(args.sac_model_template, seed)
                vecnorm_path = build_path(args.sac_vecnorm_template, seed)

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"{algo.upper()} model not found for seed {seed}: {model_path}"
                )
            if not os.path.exists(vecnorm_path):
                raise FileNotFoundError(
                    f"{algo.upper()} vecnorm not found for seed {seed}: {vecnorm_path}"
                )

            m = evaluate_one_seed(
                algo=algo,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                n_eval_episodes=args.episodes,
                render=args.render,
            )
            m["algo"] = algo
            m["seed"] = seed
            per_seed_metrics.append(m)
            seed_rows.append(m)

        seed_df = pd.DataFrame(per_seed_metrics)
        agg = aggregate_seed_metrics(seed_df)
        agg["algo"] = algo
        agg["episodes_per_seed"] = int(args.episodes)
        agg["n_seeds"] = int(len(args.seeds))
        algo_to_agg[algo] = agg
        print_human_summary(algo, agg, args.episodes, args.seeds)

    seed_metrics_df = pd.DataFrame(seed_rows)
    aggregate_df = pd.DataFrame([algo_to_agg[a] for a in args.algos])

    seed_csv = os.path.join(args.out_dir, "seed_level_metrics.csv")
    agg_csv = os.path.join(args.out_dir, "aggregate_metrics_mean_std.csv")
    agg_json = os.path.join(args.out_dir, "aggregate_metrics_mean_std.json")
    seed_metrics_df.to_csv(seed_csv, index=False)
    aggregate_df.to_csv(agg_csv, index=False)
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_df.to_dict(orient="records"), f, indent=2)

    plot_multiseed(algo_to_agg, args.out_dir)

    print(f"\nSaved seed-level metrics: {seed_csv}")
    print(f"Saved aggregate metrics: {agg_csv}")
    print(f"Saved aggregate JSON: {agg_json}")
    print(f"Saved plots (PNG + PDF): {args.out_dir}")


if __name__ == "__main__":
    main()
