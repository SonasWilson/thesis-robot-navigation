"""
Capture top-down screenshots of static_2, dynamic_2, and all four
unseen generalization environments from eval_sac_vs_td3.py.
Saves PNG files directly — no screen recording or GUI needed.

Usage:
    python screenshot_envs.py
"""

import numpy as np
import cv2
import pybullet as p

from static_2 import StaticObstacleNavEnv
from dynamic_2 import DynamicObstacleNavEnv
from curriculum_train import CurriculumObstacleNavEnv, get_stage_config, DEFAULT_STAGES
from eval_sac_vs_td3 import (
    UA_MoreStaticObstacles,
    UB_LargerArena,
    UC_FasterDynamic,
    UD_ShiftedObstacles,
)
from unseen_envs_v2 import V1TopologyV2Env, V2DynamicsV2Env, V3CombinedV2Env


def capture(env, filename: str, width: int = 800, height: int = 800):
    arena = getattr(env, "arena_size", None) or 50.0  # v2 envs use module constant

    view = p.computeViewMatrix(
        cameraEyePosition   =[arena/2, arena/2, arena * 1.5],
        cameraTargetPosition=[arena/2, arena/2, 0.0],
        cameraUpVector      =[0, 1, 0],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=55, aspect=width/height,
        nearVal=0.1, farVal=arena * 4,
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    img = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(filename, bgr)
    print(f"Saved: {filename}")


# --- Static env ---
env = StaticObstacleNavEnv()
env.reset(seed=42)
capture(env, "screenshot_static2.png")
env.close()

# --- Dynamic env (step a few times so agents are mid-motion) ---
env = DynamicObstacleNavEnv()
env.reset(seed=42)
for _ in range(120):                          # advance ~0.5 sec of sim
    env.step(env.action_space.sample())
capture(env, "screenshot_dynamic2.png")
env.close()

# --- Unseen conditions ---
unseen = [
    (UA_MoreStaticObstacles, "screenshot_UA_more_obstacles.png",  False),
    (UB_LargerArena,         "screenshot_UB_larger_arena.png",    False),
    (UC_FasterDynamic,       "screenshot_UC_faster_dynamic.png",  True),
    (UD_ShiftedObstacles,    "screenshot_UD_shifted_pos.png",     False),
]

for env_cls, filename, step_sim in unseen:
    env = env_cls()
    env.reset(seed=42)
    if step_sim:                              # step dynamic envs so agents are moving
        for _ in range(120):
            env.step(env.action_space.sample())
    capture(env, filename)
    env.close()

# --- Unseen v2 environments ---
unseen_v2 = [
    (V1TopologyV2Env,  "screenshot_V1_topology.png",  True),
    (V2DynamicsV2Env,  "screenshot_V2_dynamics.png",  True),
    (V3CombinedV2Env,  "screenshot_V3_combined.png",  True),
]

for env_cls, filename, step_sim in unseen_v2:
    env = env_cls()
    env.reset(seed=42)
    if step_sim:
        for _ in range(120):
            env.step(env.action_space.sample())
    capture(env, filename)
    env.close()

# --- Curriculum stages 1-6 ---
for stage_cfg in DEFAULT_STAGES:
    orbital_fraction = 0.5 if stage_cfg.stage_id == 6 else 0.0
    env = CurriculumObstacleNavEnv(
        num_static_obstacles=stage_cfg.num_static_obstacles,
        num_dynamic_obstacles=stage_cfg.num_dynamic_obstacles,
        arena_size=stage_cfg.arena_size,
        dynamic_speed=stage_cfg.dynamic_speed,
        orbital_fraction=orbital_fraction,
    )
    env.reset(seed=42)
    # Step so dynamic agents are mid-motion
    for _ in range(120):
        env.step(env.action_space.sample())
    capture(env, f"screenshot_stage_{stage_cfg.stage_id}.png")
    env.close()
