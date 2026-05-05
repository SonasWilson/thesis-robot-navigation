"""
Unseen generalization evaluation: SAC vs TD3
Tests both models on 4 unseen conditions derived from static_2 and dynamic_2.

Unseen conditions:
  UA — More static obstacles (4 instead of 2), same arena
  UB — Larger arena (15x15 instead of 10x10)
  UC — Faster dynamic agents (omega x2), same arena
  UD — Shifted obstacle positions (never seen during training)

Usage:
    python eval_sac_vs_td3.py \
        --sac_static_model  sac_static2_navigation_seed42.zip \
        --sac_static_vec    vecnormalize_static2_sac_seed42.pkl \
        --sac_dynamic_model sac_dynamic2_navigation_seed42.zip \
        --sac_dynamic_vec   vecnormalize_dynamic2_sac_seed42.pkl \
        --td3_static_model  td3_static2_navigation_seed42.zip \
        --td3_static_vec    vecnormalize_static2_td3_seed42.pkl \
        --td3_dynamic_model td3_dynamic2_navigation_seed42.zip \
        --td3_dynamic_vec   vecnormalize_dynamic2_td3_seed42.pkl \
        --episodes 100
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
import pybullet as p
import pybullet_data
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from static_2 import StaticObstacleNavEnv
from dynamic_2 import DynamicObstacleNavEnv


# ---------------------------------------------------------------------------
# Unseen environment variants
# ---------------------------------------------------------------------------

class UA_MoreStaticObstacles(StaticObstacleNavEnv):
    """4 static obstacles instead of 2. Same arena, same goal."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacle_specs = [
            {"pos": [4.0, 5.5, 0.5], "half_extents": [0.6, 1.0, 0.5]},
            {"pos": [6.8, 3.6, 0.5], "half_extents": [0.8, 0.5, 0.5]},
            {"pos": [3.2, 2.8, 0.5], "half_extents": [0.5, 0.7, 0.5]},  # new
            {"pos": [5.5, 7.2, 0.5], "half_extents": [0.7, 0.4, 0.5]},  # new
        ]


class UB_LargerArena(StaticObstacleNavEnv):
    """15x15 arena instead of 10x10. Goal scaled proportionally."""
    def __init__(self, **kwargs):
        kwargs["arena_size"] = 15.0
        super().__init__(**kwargs)
        self.goal_pos = np.array([11.7, 11.7, 0.5], dtype=np.float32)
        self.obstacle_specs = [
            {"pos": [6.0, 8.0, 0.5],  "half_extents": [0.6, 1.0, 0.5]},
            {"pos": [10.0, 5.5, 0.5], "half_extents": [0.8, 0.5, 0.5]},
        ]


class UC_FasterDynamic(DynamicObstacleNavEnv):
    """Dynamic agents with 2x faster omega range. Same arena, same goal."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Double the speed range — never seen during training
        self.omega_min = 0.70
        self.omega_max = 1.90
        self.dynamic_cfg = [
            {
                "center": np.array([4.8, 5.2], dtype=np.float32),
                "amp":    np.array([1.6, 0.0], dtype=np.float32),
                "omega":  1.10,
                "phase":  0.0,
                "radius": 0.45,
            },
            {
                "center": np.array([6.4, 3.8], dtype=np.float32),
                "amp":    np.array([0.0, 1.5], dtype=np.float32),
                "omega":  1.20,
                "phase":  np.pi / 2.0,
                "radius": 0.45,
            },
        ]


class UD_ShiftedObstacles(StaticObstacleNavEnv):
    """Same number of obstacles but at completely different positions."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacle_specs = [
            {"pos": [2.5, 6.0, 0.5], "half_extents": [0.5, 0.9, 0.5]},  # shifted
            {"pos": [7.5, 5.0, 0.5], "half_extents": [0.9, 0.5, 0.5]},  # shifted
        ]


# Map: condition_id -> (env_class, base_env_type, description)
UNSEEN_CONDITIONS = {
    "UA": (UA_MoreStaticObstacles, "static",  "More static obstacles (4 vs 2)"),
    "UB": (UB_LargerArena,         "static",  "Larger arena (15x15 vs 10x10)"),
    "UC": (UC_FasterDynamic,       "dynamic", "Faster dynamic agents (omega x2)"),
    "UD": (UD_ShiftedObstacles,    "static",  "Shifted obstacle positions"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sac_static_model",  type=str, default="sac_static2_navigation_seed42.zip")
    p.add_argument("--sac_static_vec",    type=str, default="vecnormalize_static2_sac_seed42.pkl")
    p.add_argument("--sac_dynamic_model", type=str, default="sac_dynamic2_navigation_seed42.zip")
    p.add_argument("--sac_dynamic_vec",   type=str, default="vecnormalize_dynamic2_sac_seed42.pkl")
    p.add_argument("--td3_static_model",  type=str, default="td3_static2_navigation_seed42.zip")
    p.add_argument("--td3_static_vec",    type=str, default="vecnormalize_static2_td3_seed42.pkl")
    p.add_argument("--td3_dynamic_model", type=str, default="td3_dynamic2_navigation_seed42.zip")
    p.add_argument("--td3_dynamic_vec",   type=str, default="vecnormalize_dynamic2_td3_seed42.pkl")
    p.add_argument("--episodes",          type=int, default=100)
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--warmup_steps",      type=int, default=500)
    p.add_argument("--device",            type=str, default="auto")
    p.add_argument("--out_csv",           type=str, default="sac_vs_td3_unseen.csv")
    return p.parse_args()


def make_vec_env(env_cls) -> DummyVecEnv:
    def _make():
        return Monitor(env_cls(), filename=None,
                       info_keywords=("success", "collision", "path_length"))
    vec = DummyVecEnv([_make])
    vec.reset()
    return vec


def warmup(vec_env: VecNormalize, steps: int):
    vec_env.training = True
    obs = vec_env.reset()
    for _ in range(steps):
        action = np.array([vec_env.action_space.sample()])
        obs, _, done, _ = vec_env.step(action)
        if done[0]:
            obs = vec_env.reset()
    vec_env.training = False


def run_eval(model, vec_env: VecNormalize,
             episodes: int, base_seed: int) -> List[Dict]:
    rows = []
    for ep in range(episodes):
        for i, env in enumerate(vec_env.venv.envs):
            env.reset(seed=base_seed + ep + i)
        obs  = vec_env.reset()
        done = [False]
        ep_r = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = vec_env.step(action)
            ep_r += float(rew[0])
        ei = info[0]
        rows.append({
            "reward":         ep_r,
            "success":        float(ei.get("success", 0) > 0.5),
            "collision":      float(ei.get("collision", 0) > 0.5),
            "episode_length": int(ei.get("episode_length", 0)),
            "path_length":    float(ei.get("path_length", 0.0)),
        })
    return rows


def summarise(rows: List[Dict], algo: str, condition: str, description: str) -> Dict:
    df = pd.DataFrame(rows)
    return {
        "algo":           algo,
        "condition":      condition,
        "description":    description,
        "success_rate":   float(df["success"].mean()),
        "collision_rate": float(df["collision"].mean()),
        "avg_reward":     float(df["reward"].mean()),
        "std_reward":     float(df["reward"].std(ddof=0)),
        "avg_ep_length":  float(df["episode_length"].mean()),
        "avg_path_length":float(df["path_length"].mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Model paths: algo -> env_type -> (model_path, vecnorm_path, model_class)
    models = {
        "SAC": {
            "static":  (args.sac_static_model,  args.sac_static_vec,  SAC),
            "dynamic": (args.sac_dynamic_model, args.sac_dynamic_vec, SAC),
        },
        "TD3": {
            "static":  (args.td3_static_model,  args.td3_static_vec,  TD3),
            "dynamic": (args.td3_dynamic_model, args.td3_dynamic_vec, TD3),
        },
    }

    all_results = []

    for condition_id, (env_cls, env_type, description) in UNSEEN_CONDITIONS.items():
        for algo, algo_models in models.items():
            model_path, vec_path, model_cls = algo_models[env_type]

            if not os.path.exists(model_path):
                print(f"  SKIP {algo} {condition_id}: model not found ({model_path})")
                continue
            if not os.path.exists(vec_path):
                print(f"  SKIP {algo} {condition_id}: vecnorm not found ({vec_path})")
                continue

            print(f"\n[{algo}] {condition_id} — {description}")

            vec_raw = make_vec_env(env_cls)
            vec_env = VecNormalize.load(vec_path, vec_raw)
            vec_env.training    = False
            vec_env.norm_reward = False

            if args.warmup_steps > 0:
                warmup(vec_env, args.warmup_steps)

            model = model_cls.load(model_path, env=vec_env, device=args.device)
            rows  = run_eval(model, vec_env, args.episodes, base_seed=args.seed)
            summary = summarise(rows, algo, condition_id, description)
            all_results.append(summary)

            print(f"  success={summary['success_rate']:.3f}  "
                  f"collision={summary['collision_rate']:.3f}  "
                  f"avg_reward={summary['avg_reward']:.1f}  "
                  f"avg_len={summary['avg_ep_length']:.0f}")
            vec_env.close()

    df = pd.DataFrame(all_results).sort_values(["condition", "algo"])

    print("\n" + "="*70)
    print("SAC vs TD3 — Unseen Generalization Summary")
    print("="*70)
    print(df[["algo","condition","description","success_rate",
              "collision_rate","avg_reward","std_reward"]].to_string(index=False))

    # Per-algorithm aggregate
    print("\n--- Aggregate (mean across all unseen conditions) ---")
    agg = df.groupby("algo")[["success_rate","collision_rate","avg_reward","std_reward"]].mean()
    print(agg.to_string())

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")

    # Recommendation
    sac_success = df[df["algo"]=="SAC"]["success_rate"].mean()
    td3_success = df[df["algo"]=="TD3"]["success_rate"].mean()
    winner = "SAC" if sac_success >= td3_success else "TD3"
    print(f"\nRecommendation: {winner} generalizes better "
          f"(SAC avg={sac_success:.3f}, TD3 avg={td3_success:.3f})")


if __name__ == "__main__":
    main()
