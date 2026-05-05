"""
generalization_assessment.py
============================
Rigorous generalization assessment for the curriculum-trained SAC navigation policy.

Three assessment modes are provided:

1. ZERO-SHOT  — evaluate the Stage-5 (or Stage-6) model on configs it was never
                explicitly trained on, without any weight updates.

2. WARM-UP    — allow a small number of fine-tuning steps on the new environment,
                then evaluate (simulates minimal adaptation cost).

3. OOD STRESS — systematically sweep obstacle counts, speeds, and arena sizes
                beyond the training distribution to find the policy's breaking point.

All results are written to CSV and JSON for thesis tables / plots.

Usage examples
--------------
# Zero-shot on all OOD configs using the Stage-5 model
python generalization_assessment.py --mode zero_shot --source_stage 5

# Warm-up on 3 custom configs (10 000 gradient steps each)
python generalization_assessment.py --mode warm_up --source_stage 5 --warmup_steps 10000

# Full OOD stress sweep (finds performance cliff)
python generalization_assessment.py --mode ood_stress --source_stage 5

# Run all three modes back-to-back
python generalization_assessment.py --mode all --source_stage 5
"""

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from curriculum_train import CurriculumObstacleNavEnv


# ---------------------------------------------------------------------------
# Out-of-distribution environment configurations
# These are deliberately outside the curriculum training distribution.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OODConfig:
    """Describes a single OOD evaluation environment."""
    config_id: str
    description: str
    num_static_obstacles: int
    num_dynamic_obstacles: int
    arena_size: float
    dynamic_speed: float
    orbital_fraction: float = 0.0   # 0 = all sinusoidal; 1 = all orbital


# ── Zero-shot configs ──────────────────────────────────────────────────────
# These probe distinct generalisation axes relative to the Stage-5 training
# distribution (3 static, 4 dynamic, 50×50, speed 1.20).

ZERO_SHOT_CONFIGS: List[OODConfig] = [
    # --- Density generalisation (more obstacles, same arena) ---
    OODConfig("ZS-D1", "High static density, same arena",
              num_static_obstacles=5, num_dynamic_obstacles=4,
              arena_size=50.0, dynamic_speed=1.20),
    OODConfig("ZS-D2", "High dynamic density, same arena",
              num_static_obstacles=3, num_dynamic_obstacles=7,
              arena_size=50.0, dynamic_speed=1.20),
    OODConfig("ZS-D3", "Mixed high density",
              num_static_obstacles=5, num_dynamic_obstacles=6,
              arena_size=50.0, dynamic_speed=1.20),

    # --- Speed generalisation (faster dynamics) ---
    OODConfig("ZS-S1", "Faster dynamics (+25%)",
              num_static_obstacles=3, num_dynamic_obstacles=4,
              arena_size=50.0, dynamic_speed=1.50),
    OODConfig("ZS-S2", "Much faster dynamics (+67%)",
              num_static_obstacles=3, num_dynamic_obstacles=4,
              arena_size=50.0, dynamic_speed=2.00),

    # --- Arena scale generalisation ---
    OODConfig("ZS-A1", "Smaller unseen arena (15×15)",
              num_static_obstacles=2, num_dynamic_obstacles=2,
              arena_size=15.0, dynamic_speed=0.85),
    OODConfig("ZS-A2", "Larger unseen arena (70×70)",
              num_static_obstacles=4, num_dynamic_obstacles=5,
              arena_size=70.0, dynamic_speed=1.20),

    # --- Motion pattern generalisation (orbital dynamics not in Stages 1-5) ---
    OODConfig("ZS-M1", "All orbital motion (fully OOD motion pattern)",
              num_static_obstacles=3, num_dynamic_obstacles=4,
              arena_size=50.0, dynamic_speed=1.20,
              orbital_fraction=1.0),
    OODConfig("ZS-M2", "Mixed orbital + sinusoidal",
              num_static_obstacles=3, num_dynamic_obstacles=4,
              arena_size=50.0, dynamic_speed=1.20,
              orbital_fraction=0.5),

    # --- Combined hard OOD ---
    OODConfig("ZS-H1", "Hard combined: high density + speed",
              num_static_obstacles=5, num_dynamic_obstacles=6,
              arena_size=50.0, dynamic_speed=1.60),
    OODConfig("ZS-H2", "Very hard: extreme density + speed",
              num_static_obstacles=6, num_dynamic_obstacles=8,
              arena_size=50.0, dynamic_speed=2.00),
]

# ── Warm-up configs ────────────────────────────────────────────────────────
# Same as ZS-D2, ZS-S2, and ZS-H1; we want to measure how much fine-tuning
# helps on the hardest axes.

WARM_UP_CONFIGS: List[OODConfig] = [
    OODConfig("WU-D", "High dynamic density (warm-up)",
              num_static_obstacles=3, num_dynamic_obstacles=7,
              arena_size=50.0, dynamic_speed=1.20),
    OODConfig("WU-S", "Much faster dynamics (warm-up)",
              num_static_obstacles=3, num_dynamic_obstacles=4,
              arena_size=50.0, dynamic_speed=2.00),
    OODConfig("WU-H", "Hard combined (warm-up)",
              num_static_obstacles=5, num_dynamic_obstacles=6,
              arena_size=50.0, dynamic_speed=1.60),
]

# ── OOD stress-sweep configs ───────────────────────────────────────────────
# Systematically increase difficulty along each axis to find the performance cliff.

def build_ood_stress_configs() -> List[OODConfig]:
    configs = []
    # Sweep: dynamic obstacle count  (3 → 12)
    for n in [3, 5, 7, 9, 12]:
        configs.append(OODConfig(
            f"ST-DYN-{n}", f"Stress: {n} dynamic obstacles",
            num_static_obstacles=3, num_dynamic_obstacles=n,
            arena_size=50.0, dynamic_speed=1.20))

    # Sweep: dynamic speed (1.20 → 3.00)
    for spd in [1.20, 1.50, 1.80, 2.20, 3.00]:
        configs.append(OODConfig(
            f"ST-SPD-{spd:.2f}", f"Stress: speed={spd:.2f}",
            num_static_obstacles=3, num_dynamic_obstacles=4,
            arena_size=50.0, dynamic_speed=spd))

    # Sweep: static obstacle count (3 → 10)
    for n in [3, 5, 7, 10]:
        configs.append(OODConfig(
            f"ST-STA-{n}", f"Stress: {n} static obstacles",
            num_static_obstacles=n, num_dynamic_obstacles=4,
            arena_size=50.0, dynamic_speed=1.20))

    # Sweep: arena size (10 → 100)
    for sz in [10.0, 25.0, 50.0, 70.0, 100.0]:
        configs.append(OODConfig(
            f"ST-SZ-{int(sz)}", f"Stress: arena={sz}",
            num_static_obstacles=3, num_dynamic_obstacles=4,
            arena_size=sz, dynamic_speed=1.20))

    # Sweep: orbital fraction (motion OOD)
    for frac in [0.0, 0.25, 0.50, 0.75, 1.00]:
        configs.append(OODConfig(
            f"ST-ORB-{int(frac*100)}", f"Stress: orbital_frac={frac:.2f}",
            num_static_obstacles=3, num_dynamic_obstacles=4,
            arena_size=50.0, dynamic_speed=1.20,
            orbital_fraction=frac))
    return configs


OOD_STRESS_CONFIGS: List[OODConfig] = build_ood_stress_configs()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(cfg: OODConfig, seed: int, render: bool = False) -> DummyVecEnv:
    def _make():
        env = CurriculumObstacleNavEnv(
            num_static_obstacles=cfg.num_static_obstacles,
            num_dynamic_obstacles=cfg.num_dynamic_obstacles,
            arena_size=cfg.arena_size,
            dynamic_speed=cfg.dynamic_speed,
            orbital_fraction=cfg.orbital_fraction,
            render_mode="human" if render else None,
        )
        env = Monitor(env, filename=None, info_keywords=("success", "collision", "path_length"))
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_make])


# ---------------------------------------------------------------------------
# Episode-level evaluation with rich per-episode logging
# ---------------------------------------------------------------------------

def run_episodes(
    model: SAC,
    vec_env: VecNormalize,
    n_episodes: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run n_episodes deterministically and return:
      - per-episode DataFrame
      - aggregate metrics dict
    """
    rows = []
    for ep in range(n_episodes):
        ep_seed = base_seed + ep * 7
        for inner_env in vec_env.venv.envs:
            inner_env.reset(seed=ep_seed)
        obs = vec_env.reset()
        done = [False]
        ep_reward = 0.0
        ep_steps = 0
        min_lidar_ep = np.inf
        path_length = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1
            if "min_lidar" in info[0]:
                min_lidar_ep = min(min_lidar_ep, float(info[0]["min_lidar"]))

        ep_info = info[0]
        success = float(ep_info.get("success", 0.0)) > 0.5
        collision = float(ep_info.get("collision", 0.0)) > 0.5
        path_length = float(ep_info.get("path_length", 0.0))
        dist_to_goal = float(ep_info.get("dist_to_goal", float("nan")))

        rows.append({
            "episode": ep + 1,
            "seed": ep_seed,
            "success": int(success),
            "collision": int(collision),
            "timeout": int(not success and not collision),
            "episode_reward": ep_reward,
            "episode_length_steps": ep_steps,
            "path_length": path_length,
            "min_lidar_distance": min_lidar_ep if min_lidar_ep < np.inf else float("nan"),
            "final_dist_to_goal": dist_to_goal,
        })

    df = pd.DataFrame(rows)
    n = len(df)

    # 95% Wilson confidence intervals for binomial proportions
    def wilson_ci(p_hat, n, z=1.96):
        denom = 1 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        return float(np.clip(centre - margin, 0, 1)), float(np.clip(centre + margin, 0, 1))

    sr = float(df["success"].mean())
    cr = float(df["collision"].mean())
    sr_lo, sr_hi = wilson_ci(sr, n)
    cr_lo, cr_hi = wilson_ci(cr, n)

    agg = {
        "n_episodes": n,
        "success_rate": sr,
        "success_rate_ci95_lo": sr_lo,
        "success_rate_ci95_hi": sr_hi,
        "collision_rate": cr,
        "collision_rate_ci95_lo": cr_lo,
        "collision_rate_ci95_hi": cr_hi,
        "timeout_rate": float(df["timeout"].mean()),
        "avg_episode_reward": float(df["episode_reward"].mean()),
        "std_episode_reward": float(df["episode_reward"].std(ddof=0)),
        "avg_episode_length": float(df["episode_length_steps"].mean()),
        "avg_path_length": float(df["path_length"].mean()),
        "avg_min_lidar": float(df["min_lidar_distance"].mean()),
        "avg_final_dist_to_goal": float(df["final_dist_to_goal"].mean()),
    }
    return df, agg


# ---------------------------------------------------------------------------
# Load model + VecNormalize from disk (frozen for evaluation)
# ---------------------------------------------------------------------------

def load_frozen_model(
    source_stage: int,
    model_dir: str,
    eval_env: DummyVecEnv,
    device: str,
) -> Tuple[SAC, VecNormalize]:
    model_path = os.path.join(model_dir, f"sac_stage_{source_stage}.zip")
    vecnorm_path = os.path.join(model_dir, f"vecnormalize_stage_{source_stage}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

    vec_env = VecNormalize.load(vecnorm_path, eval_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = SAC.load(model_path, env=vec_env, device=device)
    return model, vec_env


# ---------------------------------------------------------------------------
# MODE 1 — Zero-shot evaluation
# ---------------------------------------------------------------------------

def run_zero_shot(args) -> List[Dict]:
    print("\n" + "=" * 60)
    print("MODE: ZERO-SHOT GENERALIZATION")
    print("=" * 60)
    results = []

    for cfg in ZERO_SHOT_CONFIGS:
        print(f"\n[Zero-Shot] {cfg.config_id}: {cfg.description}")
        raw_env = make_env(cfg, seed=args.seed, render=args.render)

        try:
            model, vec_env = load_frozen_model(
                source_stage=args.source_stage,
                model_dir=args.model_dir,
                eval_env=raw_env,
                device=args.device,
            )
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            raw_env.close()
            continue

        _, agg = run_episodes(
            model=model,
            vec_env=vec_env,
            n_episodes=args.episodes,
            base_seed=args.seed,
        )
        vec_env.close()

        row = {"mode": "zero_shot", "source_stage": args.source_stage}
        row.update(asdict(cfg))
        row.update(agg)
        results.append(row)
        _print_row(row)

    return results


# ---------------------------------------------------------------------------
# MODE 2 — Warm-up fine-tuning evaluation
# ---------------------------------------------------------------------------

def run_warm_up(args) -> List[Dict]:
    print("\n" + "=" * 60)
    print("MODE: WARM-UP GENERALIZATION")
    print("=" * 60)
    results = []

    for cfg in WARM_UP_CONFIGS:
        print(f"\n[Warm-Up] {cfg.config_id}: {cfg.description}")

        # ── Zero-shot baseline ──
        raw_env_zs = make_env(cfg, seed=args.seed, render=False)
        try:
            model_zs, vec_env_zs = load_frozen_model(
                source_stage=args.source_stage,
                model_dir=args.model_dir,
                eval_env=raw_env_zs,
                device=args.device,
            )
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            raw_env_zs.close()
            continue

        _, agg_zs = run_episodes(
            model=model_zs,
            vec_env=vec_env_zs,
            n_episodes=args.episodes,
            base_seed=args.seed,
        )
        vec_env_zs.close()

        row_zs = {"mode": "warm_up_zero_shot", "source_stage": args.source_stage,
                  "warmup_steps": 0}
        row_zs.update(asdict(cfg))
        row_zs.update(agg_zs)
        results.append(row_zs)
        print(f"  [Before warm-up] success={agg_zs['success_rate']:.3f}, "
              f"collision={agg_zs['collision_rate']:.3f}")

        # ── Fine-tune (warm-up) ──
        raw_env_wu = make_env(cfg, seed=args.seed + 1000, render=False)
        model_path = os.path.join(args.model_dir, f"sac_stage_{args.source_stage}.zip")
        vecnorm_path = os.path.join(args.model_dir,
                                    f"vecnormalize_stage_{args.source_stage}.pkl")

        vec_env_wu = VecNormalize.load(vecnorm_path, raw_env_wu)
        vec_env_wu.training = True    # re-enable norm updates during warm-up
        vec_env_wu.norm_reward = False

        model_wu = SAC.load(model_path, env=vec_env_wu, device=args.device)

        print(f"  Fine-tuning for {args.warmup_steps} steps ...")
        t0 = time.time()
        model_wu.learn(total_timesteps=args.warmup_steps, reset_num_timesteps=False)
        print(f"  Fine-tuning done in {time.time() - t0:.1f}s")

        vec_env_wu.training = False

        # Evaluate after warm-up
        raw_env_eval = make_env(cfg, seed=args.seed + 2000, render=args.render)
        vec_env_eval = VecNormalize.load(vecnorm_path, raw_env_eval)
        vec_env_eval.obs_rms = copy.deepcopy(vec_env_wu.obs_rms)
        vec_env_eval.training = False
        vec_env_eval.norm_reward = False
        model_wu.set_env(vec_env_eval)

        _, agg_wu = run_episodes(
            model=model_wu,
            vec_env=vec_env_eval,
            n_episodes=args.episodes,
            base_seed=args.seed + 2000,
        )
        vec_env_wu.close()
        vec_env_eval.close()

        row_wu = {"mode": "warm_up_after", "source_stage": args.source_stage,
                  "warmup_steps": args.warmup_steps}
        row_wu.update(asdict(cfg))
        row_wu.update(agg_wu)
        results.append(row_wu)
        print(f"  [After  warm-up] success={agg_wu['success_rate']:.3f}, "
              f"collision={agg_wu['collision_rate']:.3f}")

        delta_sr = agg_wu["success_rate"] - agg_zs["success_rate"]
        print(f"  Δ success rate from warm-up: {delta_sr:+.3f}")

    return results


# ---------------------------------------------------------------------------
# MODE 3 — OOD stress sweep (finds the performance cliff)
# ---------------------------------------------------------------------------

def run_ood_stress(args) -> List[Dict]:
    print("\n" + "=" * 60)
    print("MODE: OOD STRESS SWEEP")
    print("=" * 60)
    results = []

    for cfg in OOD_STRESS_CONFIGS:
        print(f"\n[OOD Stress] {cfg.config_id}: {cfg.description}")
        raw_env = make_env(cfg, seed=args.seed, render=False)

        try:
            model, vec_env = load_frozen_model(
                source_stage=args.source_stage,
                model_dir=args.model_dir,
                eval_env=raw_env,
                device=args.device,
            )
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            raw_env.close()
            continue

        _, agg = run_episodes(
            model=model,
            vec_env=vec_env,
            n_episodes=args.stress_episodes,
            base_seed=args.seed,
        )
        vec_env.close()

        row = {"mode": "ood_stress", "source_stage": args.source_stage}
        row.update(asdict(cfg))
        row.update(agg)
        results.append(row)
        _print_row(row)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_row(row: Dict):
    print(
        f"  success={row['success_rate']:.3f} "
        f"[{row['success_rate_ci95_lo']:.3f}, {row['success_rate_ci95_hi']:.3f}] | "
        f"collision={row['collision_rate']:.3f} | "
        f"timeout={row['timeout_rate']:.3f} | "
        f"avg_reward={row['avg_episode_reward']:.2f}"
    )


def save_results(results: List[Dict], out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"generalization_{tag}.csv")
    json_path = os.path.join(out_dir, f"generalization_{tag}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    return df


def print_summary_table(df: pd.DataFrame, title: str):
    cols = [
        "config_id", "description",
        "num_static_obstacles", "num_dynamic_obstacles",
        "arena_size", "dynamic_speed", "orbital_fraction",
        "success_rate", "success_rate_ci95_lo", "success_rate_ci95_hi",
        "collision_rate", "timeout_rate",
        "avg_episode_reward", "avg_episode_length",
    ]
    available = [c for c in cols if c in df.columns]
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df[available].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generalization assessment for curriculum-trained SAC navigation policy"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "warm_up", "ood_stress", "all"],
        help="Assessment mode (default: zero_shot)"
    )
    parser.add_argument(
        "--source_stage",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5, 6],
        help="Which trained stage model to assess (default: 5)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory containing sac_stage_*.zip and vecnormalize_stage_*.pkl"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./generalization_results",
        help="Directory to write CSV/JSON results"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per config for zero-shot and warm-up modes (default: 100)"
    )
    parser.add_argument(
        "--stress_episodes",
        type=int,
        default=50,
        help="Episodes per config for OOD stress sweep (default: 50, many configs)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10_000,
        help="Gradient steps for warm-up fine-tuning (default: 10 000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Base random seed — kept separate from training seed 42 (default: 99)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto | cpu | cuda (default: auto)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the first environment in GUI mode (debugging only)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    all_results: List[Dict] = []

    if args.mode in ("zero_shot", "all"):
        zs_results = run_zero_shot(args)
        all_results.extend(zs_results)
        if zs_results:
            df_zs = save_results(zs_results, args.out_dir, "zero_shot")
            print_summary_table(df_zs, "ZERO-SHOT SUMMARY")

    if args.mode in ("warm_up", "all"):
        wu_results = run_warm_up(args)
        all_results.extend(wu_results)
        if wu_results:
            df_wu = save_results(wu_results, args.out_dir, "warm_up")
            print_summary_table(df_wu, "WARM-UP SUMMARY")

    if args.mode in ("ood_stress", "all"):
        st_results = run_ood_stress(args)
        all_results.extend(st_results)
        if st_results:
            df_st = save_results(st_results, args.out_dir, "ood_stress")
            print_summary_table(df_st, "OOD STRESS SUMMARY")

    if args.mode == "all" and all_results:
        save_results(all_results, args.out_dir, "combined")
        print("\nAll modes complete. Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()
