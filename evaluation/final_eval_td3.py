import argparse
import json
import os

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from v5 import SafetySphereNavEnv


def make_env(render_mode=None):
    return SafetySphereNavEnv(render_mode=render_mode)


def run_evaluation(model_path, vecnorm_path, n_eval_episodes, render=False):
    render_mode = "human" if render else None
    eval_env = DummyVecEnv([lambda: make_env(render_mode=render_mode)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = TD3.load(model_path, env=eval_env, device="auto")

    episode_rewards = []
    episode_lengths = []
    episode_path_lengths = []
    episode_final_dist = []
    successes = 0
    collisions = 0
    timeouts = 0

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = [False]
        episode_return = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_return += float(reward[0])

        episode_info = info[0]
        episode_rewards.append(float(episode_return))
        episode_lengths.append(int(episode_info.get("episode_length", 0)))
        episode_path_lengths.append(float(episode_info.get("path_length", 0.0)))
        episode_final_dist.append(float(episode_info.get("dist_to_goal", 0.0)))

        success = float(episode_info.get("success", 0.0))
        collision = float(episode_info.get("collision", 0.0))
        successes += int(success > 0.5)
        collisions += int(collision > 0.5)
        if success < 0.5 and collision < 0.5:
            timeouts += 1

    eval_env.close()

    metrics = {
        "n_eval_episodes": int(n_eval_episodes),
        "success_rate": float(successes / n_eval_episodes),
        "collision_rate": float(collisions / n_eval_episodes),
        "timeout_rate": float(timeouts / n_eval_episodes),
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "avg_episode_length_steps": float(np.mean(episode_lengths)),
        "avg_path_length": float(np.mean(episode_path_lengths)),
        "avg_final_dist_to_goal": float(np.mean(episode_final_dist)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Final TD3 policy evaluation")
    parser.add_argument("--model_path", type=str, default="td3_navigation.zip")
    parser.add_argument("--vecnorm_path", type=str, default="vecnormalize_td3.pkl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=str, default="./logs_td3/final_eval_metrics.json")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.vecnorm_path):
        raise FileNotFoundError(f"VecNormalize stats not found: {args.vecnorm_path}")

    metrics = run_evaluation(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        n_eval_episodes=args.episodes,
        render=args.render,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=== Final Evaluation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved metrics to: {args.out}")


if __name__ == "__main__":
    main()
