from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from v5 import SafetySphereNavEnv

import torch.nn as nn
import os

log_dir = "./tb_logs_sac/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("./checkpoints_sac/", exist_ok=True)
os.makedirs("./logs_sac/", exist_ok=True)

SEED = 42


def make_env(curriculum_progress=1.0):
    return Monitor(
        SafetySphereNavEnv(curriculum_progress=curriculum_progress),
        filename="./logs_sac/train_monitor.csv",
        info_keywords=("success", "collision", "path_length"),
    )


def make_eval_env():
    return Monitor(
        SafetySphereNavEnv(curriculum_progress=1.0),
        filename="./logs_sac/eval_monitor.csv",
        info_keywords=("success", "collision", "path_length"),
    )


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_steps=600_000, verbose=0):
        super().__init__(verbose=verbose)
        self.curriculum_steps = int(curriculum_steps)

    def _on_step(self):
        t = self.num_timesteps
        curriculum_ratio = min(t / max(self.curriculum_steps, 1), 1.0)
        vecnorm_env = self.model.get_env()
        for env in vecnorm_env.venv.envs:
            if hasattr(env, "env") and hasattr(env.env, "set_curriculum_progress"):
                env.env.set_curriculum_progress(curriculum_ratio)

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
    save_path="./checkpoints_sac/",
    name_prefix="sac_nav",
    save_replay_buffer=True
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_sac/",
    log_path="./logs_sac/",
    eval_freq=20_000,
    n_eval_episodes=20
)

curriculum_callback = CurriculumCallback(curriculum_steps=600_000)
callback = CallbackList([checkpoint_callback, eval_callback, curriculum_callback])

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=20_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    target_entropy="auto",
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

train_env.save("vecnormalize_sac.pkl")
model.save("sac_navigation")
