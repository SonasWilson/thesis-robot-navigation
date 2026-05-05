from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from sphere_nav_env import SphereNavEnv
import numpy as np

env = SphereNavEnv()
n_actions = env.action_space.shape[-1]

noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=noise, verbose=1, learning_rate=1e-3, tensorboard_log="./tb_logs/")

model.learn(total_timesteps=50000)

model.save("ddpg_gap1_baseline")
env.close()