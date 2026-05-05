import argparse
import os

import numpy as np
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from static_2 import StaticObstacleNavEnv


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 training for static 2-obstacle navigation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--arena_size", type=float, default=10.0)
    parser.add_argument("--log_dir", type=str, default="./logs_static2_td3")
    parser.add_argument("--tb_dir", type=str, default="./tb_logs_static2_td3")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_static2_td3")

    # Continuous training inputs (for stage-to-stage transfer).
    parser.add_argument("--resume_model_path", type=str, default="")
    parser.add_argument("--resume_vecnorm_path", type=str, default="")

    # Output names (use seed naming for multi-run experiments).
    parser.add_argument("--model_out", type=str, default="td3_static2_navigation_seed{seed}")
    parser.add_argument("--vecnorm_out", type=str, default="vecnormalize_static2_td3_seed{seed}.pkl")
    return parser.parse_args()


def build_train_env(arena_size: float, log_dir: str):
    def _make():
        return Monitor(
            StaticObstacleNavEnv(arena_size=arena_size),
            filename=os.path.join(log_dir, "train_monitor.csv"),
            info_keywords=("success", "collision", "path_length"),
        )

    return DummyVecEnv([_make])


def build_eval_env(arena_size: float, log_dir: str):
    def _make():
        return Monitor(
            StaticObstacleNavEnv(arena_size=arena_size),
            filename=os.path.join(log_dir, "eval_monitor.csv"),
            info_keywords=("success", "collision", "path_length"),
        )

    return DummyVecEnv([_make])


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    raw_train_env = build_train_env(args.arena_size, args.log_dir)
    raw_eval_env = build_eval_env(args.arena_size, args.log_dir)

    if args.resume_vecnorm_path:
        train_env = VecNormalize.load(args.resume_vecnorm_path, raw_train_env)
        train_env.training = True
        train_env.norm_reward = False
    else:
        train_env = VecNormalize(raw_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = VecNormalize(raw_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=args.ckpt_dir,
        name_prefix="td3_static2",
        save_replay_buffer=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.log_dir,
        log_path=args.log_dir,
        eval_freq=20_000,
        n_eval_episodes=20,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    n_actions = train_env.action_space.shape[-1]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.20 * np.ones(n_actions))

    if args.resume_model_path:
        model = TD3.load(args.resume_model_path, env=train_env, device="auto")
        model.action_noise = noise
        model.tensorboard_log = args.tb_dir
    else:
        model = TD3(
            "MlpPolicy",
            train_env,
            action_noise=noise,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=20_000,
            batch_size=128,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ReLU),
            seed=args.seed,
            verbose=1,
            tensorboard_log=args.tb_dir,
            device="auto",
        )

    model.learn(total_timesteps=args.timesteps, callback=callback, log_interval=10, progress_bar=True)

    model_out = args.model_out.format(seed=args.seed)
    vec_out = args.vecnorm_out.format(seed=args.seed)
    train_env.save(vec_out)
    model.save(model_out)

    print(f"Saved model: {model_out}.zip")
    print(f"Saved vecnorm: {vec_out}")


if __name__ == "__main__":
    main()
