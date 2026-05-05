import argparse
import json
import os
from typing import Dict, List

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from curriculum_train import DEFAULT_STAGES, CurriculumObstacleNavEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all curriculum SAC stage models")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--out_csv", type=str, default="./models/curriculum_eval_summary.csv")
    parser.add_argument("--out_json", type=str, default="./models/curriculum_eval_summary.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./logs", help="Root log directory used during curriculum training")
    parser.add_argument("--stage_start", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--stage_end", type=int, default=5, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def make_eval_env(stage_cfg, seed: int, render: bool = False):
    def _make():
        orbital_fraction = 0.5 if stage_cfg.stage_id == 6 else 0.0
        env = CurriculumObstacleNavEnv(
            num_static_obstacles=stage_cfg.num_static_obstacles,
            num_dynamic_obstacles=stage_cfg.num_dynamic_obstacles,
            arena_size=stage_cfg.arena_size,
            dynamic_speed=stage_cfg.dynamic_speed,
            orbital_fraction=orbital_fraction,
            render_mode="human" if render else None,
        )
        env = Monitor(env, filename=None, info_keywords=("success", "collision", "path_length"))
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_make])


def read_stage_gate_artifacts(stage_id: int, log_dir: str) -> Dict:
    stage_dir = os.path.join(log_dir, f"stage_{stage_id}")
    summary_path = os.path.join(stage_dir, "stage_gate_summary.json")
    gate_log_path = os.path.join(stage_dir, "stage_gate_log.csv")

    out = {
        "gate_completed": None,
        "gate_train_steps": None,
        "gate_eval_count": None,
        "last_gate_decision": None,
        "last_gate_success_rate": None,
        "last_gate_collision_rate": None,
        "last_gate_reward_mean": None,
        "last_gate_reward_std": None,
        "last_gate_episode_length_mean": None,
    }

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        out["gate_completed"] = summary.get("completed")
        out["gate_train_steps"] = summary.get("train_steps")
        out["gate_eval_count"] = summary.get("eval_count")

    if os.path.exists(gate_log_path):
        gate_df = pd.read_csv(gate_log_path)
        if not gate_df.empty:
            last = gate_df.iloc[-1]
            out["last_gate_decision"] = str(last.get("decision", ""))
            out["last_gate_success_rate"] = float(last.get("success_rate"))
            out["last_gate_collision_rate"] = float(last.get("collision_rate"))
            out["last_gate_reward_mean"] = float(last.get("reward_mean"))
            out["last_gate_reward_std"] = float(last.get("reward_std"))
            out["last_gate_episode_length_mean"] = float(last.get("episode_length_mean"))

    return out


def evaluate_stage(stage_cfg, args) -> Dict:
    model_path = os.path.join(args.model_dir, f"sac_stage_{stage_cfg.stage_id}.zip")
    vecnorm_path = os.path.join(args.model_dir, f"vecnormalize_stage_{stage_cfg.stage_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing stage model: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"Missing stage vecnorm: {vecnorm_path}")

    env = make_eval_env(stage_cfg, seed=args.seed + stage_cfg.stage_id * 13, render=args.render)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = SAC.load(model_path, env=env, device=args.device)

    rows: List[Dict] = []
    for ep in range(args.episodes):
        # Unique seed per episode — different goal jitter, obstacle phases, etc.
        for inner_env in env.venv.envs:
            inner_env.reset(seed=args.seed + stage_cfg.stage_id * 13 + ep)
        obs = env.reset()
        done = [False]
        episode_reward = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += float(reward[0])

        ep_info = info[0]
        success = float(ep_info.get("success", 0.0))
        collision = float(ep_info.get("collision", 0.0))
        rows.append(
            {
                "stage_id": stage_cfg.stage_id,
                "episode": ep + 1,
                "episode_reward": episode_reward,
                "episode_length_steps": int(ep_info.get("episode_length", 0)),
                "success": float(success > 0.5),
                "collision": float(collision > 0.5),
            }
        )

    env.close()
    df = pd.DataFrame(rows)
    result = {
        "stage_id": stage_cfg.stage_id,
        "num_static_obstacles": stage_cfg.num_static_obstacles,
        "num_dynamic_obstacles": stage_cfg.num_dynamic_obstacles,
        "arena_size": stage_cfg.arena_size,
        "dynamic_speed": stage_cfg.dynamic_speed,
        "n_eval_episodes": int(args.episodes),
        "success_rate": float(df["success"].mean()),
        "collision_rate": float(df["collision"].mean()),
        "avg_episode_length": float(df["episode_length_steps"].mean()),
        "avg_reward": float(df["episode_reward"].mean()),
        "std_reward": float(df["episode_reward"].std(ddof=0)),
    }
    result.update(read_stage_gate_artifacts(stage_id=stage_cfg.stage_id, log_dir=args.log_dir))
    return result


def main():
    args = parse_args()
    if args.stage_start > args.stage_end:
        raise ValueError("--stage_start must be <= --stage_end")

    stage_cfgs = [cfg for cfg in DEFAULT_STAGES if args.stage_start <= cfg.stage_id <= args.stage_end]
    results = []
    for cfg in stage_cfgs:
        print(f"Evaluating Stage {cfg.stage_id} ...")
        result = evaluate_stage(cfg, args)
        results.append(result)
        print(
            f"Stage {cfg.stage_id} | success={result['success_rate']:.3f}, "
            f"collision={result['collision_rate']:.3f}, "
            f"avg_len={result['avg_episode_length']:.1f}, avg_reward={result['avg_reward']:.2f}, "
            f"gate_decision={result.get('last_gate_decision')}, "
            f"gate_completed={result.get('gate_completed')}"
        )

    summary_df = pd.DataFrame(results).sort_values("stage_id")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print("\n=== Curriculum Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved CSV: {args.out_csv}")
    print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
