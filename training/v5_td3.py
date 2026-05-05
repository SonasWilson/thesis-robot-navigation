from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from v5 import SafetySphereNavEnv

import numpy as np
import torch.nn as nn
import os

log_dir = "./tb_logs_td3/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./checkpoints_td3/", exist_ok=True)
os.makedirs("./logs_td3/", exist_ok=True)

SEED = 42


def make_env(curriculum_progress=1.0):
    return Monitor(
        SafetySphereNavEnv(curriculum_progress=curriculum_progress),
        filename="./logs_td3/train_monitor.csv",
        info_keywords=("success", "collision", "path_length"),
    )


def make_eval_env():
    return Monitor(
        SafetySphereNavEnv(curriculum_progress=1.0),
        filename="./logs_td3/eval_monitor.csv",
        info_keywords=("success", "collision", "path_length"),
    )


class AdaptiveExplorationCurriculumCallback(BaseCallback):
    def __init__(
        self,
        initial_sigma=0.20,
        final_sigma=0.04,
        sigma_decay_steps=700_000,
        curriculum_steps=600_000,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.initial_sigma = float(initial_sigma)
        self.final_sigma = float(final_sigma)
        self.sigma_decay_steps = int(sigma_decay_steps)
        self.curriculum_steps = int(curriculum_steps)

    def _on_step(self):
        t = self.num_timesteps

        # Linearly decay exploration noise to reduce oscillatory late behavior
        decay_ratio = min(t / max(self.sigma_decay_steps, 1), 1.0)
        sigma_now = self.initial_sigma + (self.final_sigma - self.initial_sigma) * decay_ratio
        # SB3 compatibility: some versions expose "sigma", others "_sigma"
        if hasattr(self.model.action_noise, "sigma"):
            self.model.action_noise.sigma = sigma_now * np.ones_like(self.model.action_noise.sigma)
        elif hasattr(self.model.action_noise, "_sigma"):
            self.model.action_noise._sigma = sigma_now * np.ones_like(self.model.action_noise._sigma)

        # Ramp difficulty from easy task to full navigation task
        curriculum_ratio = min(t / max(self.curriculum_steps, 1), 1.0)
        vecnorm_env = self.model.get_env()
        for env in vecnorm_env.venv.envs:
            if hasattr(env, "env") and hasattr(env.env, "set_curriculum_progress"):
                env.env.set_curriculum_progress(curriculum_ratio)

        self.logger.record("train/action_noise_sigma", sigma_now)
        self.logger.record("train/curriculum_progress", curriculum_ratio)
        return True


train_env = DummyVecEnv([lambda: make_env(curriculum_progress=0.0)])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    training=False
)
eval_env.obs_rms = train_env.obs_rms

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints_td3/",
    name_prefix="td3_nav",
    save_replay_buffer=True
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_td3/",
    log_path="./logs_td3/",
    eval_freq=20_000,
    n_eval_episodes=20
)

adapt_callback = AdaptiveExplorationCurriculumCallback(
    initial_sigma=0.20,
    final_sigma=0.04,
    sigma_decay_steps=700_000,
    curriculum_steps=600_000,
)
callback = CallbackList([checkpoint_callback, eval_callback, adapt_callback])

n_actions = train_env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.20 * np.ones(n_actions))

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
    seed=SEED,
    verbose=1,
    tensorboard_log=log_dir,
    device="auto"
)

model.learn(
    total_timesteps=1_500_000,
    callback=callback,
    log_interval=10,
    progress_bar=True
)

train_env.save("vecnormalize_td3.pkl")
model.save("td3_navigation")