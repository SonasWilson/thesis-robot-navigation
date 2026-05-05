import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@dataclass(frozen=True)
class StageConfig:
    stage_id: int
    num_static_obstacles: int
    num_dynamic_obstacles: int
    arena_size: float
    dynamic_speed: float


DEFAULT_STAGES: List[StageConfig] = [
    StageConfig(stage_id=1, num_static_obstacles=2, num_dynamic_obstacles=1, arena_size=10.0, dynamic_speed=0.45),
    StageConfig(stage_id=2, num_static_obstacles=2, num_dynamic_obstacles=2, arena_size=20.0, dynamic_speed=0.65),
    StageConfig(stage_id=3, num_static_obstacles=2, num_dynamic_obstacles=3, arena_size=30.0, dynamic_speed=0.85),
    StageConfig(stage_id=4, num_static_obstacles=3, num_dynamic_obstacles=3, arena_size=40.0, dynamic_speed=1.00),
    StageConfig(stage_id=5, num_static_obstacles=3, num_dynamic_obstacles=4, arena_size=50.0, dynamic_speed=1.20),
    # Stage 6: same arena as Stage 5, one extra static obstacle, one extra dynamic agent,
    # and higher dynamic speed — a natural difficulty step beyond Stage 5.
    StageConfig(stage_id=6, num_static_obstacles=4, num_dynamic_obstacles=5, arena_size=50.0, dynamic_speed=1.40),
]

# (min_steps_before_gate_checks, max_steps_budget)
STAGE_TIMESTEPS: Dict[int, Tuple[int, int]] = {
    1: (300_000,   500_000),
    2: (500_000,   800_000),
    3: (800_000,   1_200_000),
    4: (1_200_000, 1_800_000),
    5: (1_500_000, 2_500_000),
    6: (1_800_000, 3_000_000),  # Stage 6: longer budget for harder task
}


def compute_goal(arena_size: float, rng=None, jitter_ratio: float = 0.0) -> np.ndarray:
    """
    Compute curriculum-consistent goal from arena size.
    Base goal is fixed at 0.8 * arena_size for x/y.
    Optional jitter keeps goal in [0.75, 0.85] * arena_size when jitter_ratio > 0.
    """
    arena_size = float(arena_size)
    base = 0.8 * arena_size
    if rng is None or jitter_ratio <= 0.0:
        gx = base
        gy = base
    else:
        span = 0.05 * arena_size * float(np.clip(jitter_ratio, 0.0, 1.0))
        gx = float(rng.uniform(base - span, base + span))
        gy = float(rng.uniform(base - span, base + span))

    # Keep goal inside valid arena bounds.
    margin = 0.6
    gx = float(np.clip(gx, margin, arena_size - margin))
    gy = float(np.clip(gy, margin, arena_size - margin))
    return np.array([gx, gy, 0.5], dtype=np.float32)


class CurriculumObstacleNavEnv(gym.Env):
    """
    Configurable version of the existing static/dynamic navigation environments.
    Reward design keeps goal progress dominant while balancing safety and smooth motion.
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        num_static_obstacles: int,
        num_dynamic_obstacles: int,
        arena_size: float,
        dynamic_speed: float,
        goal_jitter_ratio: float = 0.0,
        orbital_fraction: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.num_static_obstacles = int(num_static_obstacles)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.arena_size = float(arena_size)
        self.dynamic_speed = float(dynamic_speed)
        self.goal_jitter_ratio = float(goal_jitter_ratio)
        # Fraction of dynamic agents that use 2D orbital motion instead of
        # 1D sinusoidal oscillation. 0.0 = all sinusoidal (Stages 1-5).
        # 0.5 = half orbital, half sinusoidal (Stage 6).
        self.orbital_fraction = float(np.clip(orbital_fraction, 0.0, 1.0))

        self.max_steps = 1800
        self.physics_substeps = 6
        self.lidar_range = max(10.0, self.arena_size * 0.35)
        self.dt = 1.0 / 240.0
        self.sim_time = 0.0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        self.start_pos = np.array([1.2, 1.2, 0.26], dtype=np.float32)
        self.goal_pos = compute_goal(self.arena_size)
        self.static_specs: List[Dict] = []
        self.dynamic_cfg: List[Dict] = []
        self.omega_min = 0.25 * self.dynamic_speed
        self.omega_max = 0.80 * self.dynamic_speed
        self.omega_update_interval = (1.2, 3.0)
        self.omega_smoothing = 0.04

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, "client"):
            try:
                p.disconnect(self.client)
            except:
                pass

        self.client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=max(12.0, self.arena_size * 0.8),
                cameraYaw=0.0,
                cameraPitch=-89.0,
                cameraTargetPosition=[self.arena_size / 2.0, self.arena_size / 2.0, 0.0],
                physicsClientId=self.client,
            )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        self.sim_time = 0.0

        # Recompute goal every reset using seeded RNG; defaults to fixed 0.8 * arena_size.
        self.goal_pos = compute_goal(self.arena_size, rng=self.np_random, jitter_ratio=self.goal_jitter_ratio)

        # Sample start position from the bottom-left region each episode.
        # Region is 10-20% of arena_size so the robot always starts near the
        # corner but with enough variation to prevent route memorisation.
        margin = 0.6
        start_hi = float(np.clip(0.15 * self.arena_size, 1.5, 6.0))
        rng = self.np_random
        for _ in range(200):
            sx = float(rng.uniform(margin, start_hi))
            sy = float(rng.uniform(margin, start_hi))
            if float(np.linalg.norm(np.array([sx, sy]) - self.goal_pos[:2])) > 3.0:
                break
        self.start_pos = np.array([sx, sy, 0.26], dtype=np.float32)

        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.wall_ids = self._create_walls()
        self.static_specs = self._sample_static_specs()
        self.static_ids = self._create_static_obstacles()
        self.dynamic_cfg = self._sample_dynamic_cfg()
        self.dynamic_ids = self._create_dynamic_obstacles()
        self._create_goal_visual()
        self.robot = self._create_robot()

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        self.prev_xy = np.array(robot_pos[:2], dtype=np.float32)
        self.prev_dist = float(np.linalg.norm(self.prev_xy - self.goal_pos[:2]))
        self.path_length = 0.0
        self.current_step = 0

        return self._get_obs(), {}

    def _sample_static_specs(self) -> List[Dict]:
        specs = []
        rng = self.np_random
        for _ in range(self.num_static_obstacles):
            hx = float(rng.uniform(0.45, 1.10))
            hy = float(rng.uniform(0.45, 1.10))
            pos = self._sample_xy(min_dist=2.0, margin=max(hx, hy) + 0.8)
            specs.append({"pos": [float(pos[0]), float(pos[1]), 0.5], "half_extents": [hx, hy, 0.5]})
        return specs

    def _sample_dynamic_cfg(self) -> List[Dict]:
        cfgs: List[Dict] = []
        rng = self.np_random
        # Determine how many agents use orbital vs sinusoidal motion.
        # Orbital agents are assigned to the first N slots so the split
        # is deterministic given orbital_fraction.
        n_orbital = int(round(self.num_dynamic_obstacles * self.orbital_fraction))
        for i in range(self.num_dynamic_obstacles):
            center = self._sample_xy(min_dist=2.2, margin=1.2)
            radius = float(rng.uniform(0.35, 0.55))
            if i < n_orbital:
                # 2D elliptical orbit: x = cx + amp_x*cos(theta)
                #                      y = cy + amp_y*sin(theta)
                # amp_x and amp_y are independent so the orbit is elliptical.
                amp_x = float(rng.uniform(0.8, 1.8))
                amp_y = float(rng.uniform(0.8, 1.8))
                omega = float(rng.uniform(self.omega_min, self.omega_max))
                cfgs.append({
                    "motion":  "orbital",
                    "center":  center.astype(np.float32),
                    "amp_x":   amp_x,
                    "amp_y":   amp_y,
                    "omega":   omega,
                    "theta":   float(rng.uniform(0.0, 2.0 * np.pi)),
                    "radius":  radius,
                })
            else:
                # 1D sinusoidal oscillation along a random axis (Stages 1-5 behaviour)
                amp = rng.uniform(0.0, 1.8, size=2).astype(np.float32)
                if float(np.linalg.norm(amp)) < 0.5:
                    amp[int(rng.integers(0, 2))] = float(rng.uniform(0.9, 1.8))
                cfgs.append({
                    "motion":       "sinusoidal",
                    "center":       center.astype(np.float32),
                    "amp":          amp,
                    "omega":        float(rng.uniform(self.omega_min, self.omega_max)),
                    "omega_target": float(rng.uniform(self.omega_min, self.omega_max)),
                    "phase":        float(rng.uniform(0.0, 2.0 * np.pi)),
                    "theta":        float(rng.uniform(0.0, 2.0 * np.pi)),
                    "radius":       radius,
                    "next_omega_update": float(rng.uniform(*self.omega_update_interval)),
                })
        return cfgs

    def _sample_xy(self, min_dist: float, margin: float) -> np.ndarray:
        rng = self.np_random
        lo = margin
        hi = self.arena_size - margin
        for _ in range(200):
            xy = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)], dtype=np.float32)
            if np.linalg.norm(xy - self.start_pos[:2]) < min_dist:
                continue
            if np.linalg.norm(xy - self.goal_pos[:2]) < min_dist:
                continue
            return xy
        return np.array([self.arena_size * 0.5, self.arena_size * 0.5], dtype=np.float32)

    def _create_walls(self):
        wall_ids = []
        thickness = 0.2
        h = 2.0
        a = self.arena_size
        walls = [
            ([-0.1, a / 2, h / 2], [thickness, a, h]),
            ([a + 0.1, a / 2, h / 2], [thickness, a, h]),
            ([a / 2, -0.1, h / 2], [a, thickness, h]),
            ([a / 2, a + 0.1, h / 2], [a, thickness, h]),
        ]
        for pos, size in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=self.client)
            wall_ids.append(p.createMultiBody(0, col, vis, basePosition=pos, physicsClientId=self.client))
        return wall_ids

    def _create_static_obstacles(self):
        ids = []
        for spec in self.static_specs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=spec["half_extents"], physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=spec["half_extents"], rgbaColor=[0.9, 0.6, 0.1, 1.0], physicsClientId=self.client)
            ids.append(p.createMultiBody(0, col, vis, basePosition=spec["pos"], physicsClientId=self.client))
        return ids

    def _create_dynamic_obstacles(self):
        ids = []
        for cfg in self.dynamic_cfg:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cfg["radius"], height=1.0, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cfg["radius"], length=1.0, rgbaColor=[1.0, 0.5, 0.0, 1.0], physicsClientId=self.client)
            x, y = cfg["center"]
            ids.append(p.createMultiBody(0, col, vis, basePosition=[float(x), float(y), 0.5], physicsClientId=self.client))
        return ids

    def _create_goal_visual(self):
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.35, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
        p.createMultiBody(0, baseVisualShapeIndex=goal_vis, basePosition=self.goal_pos, physicsClientId=self.client)

    def _create_robot(self):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0, 0, 1, 1], physicsClientId=self.client)
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self.start_pos.tolist(),
            physicsClientId=self.client,
        )
        p.changeDynamics(robot_id, -1, linearDamping=0.05, angularDamping=0.05, lateralFriction=0.8, physicsClientId=self.client)
        return robot_id

    def _update_dynamic_obstacles(self):
        rng = self.np_random
        for obs_id, cfg in zip(self.dynamic_ids, self.dynamic_cfg):
            if cfg["motion"] == "orbital":
                # 2D elliptical orbit — theta advances at constant omega
                cfg["theta"] += cfg["omega"] * self.dt
                x = cfg["center"][0] + cfg["amp_x"] * float(np.cos(cfg["theta"]))
                y = cfg["center"][1] + cfg["amp_y"] * float(np.sin(cfg["theta"]))
                xy = np.clip(np.array([x, y], dtype=np.float32),
                             0.8, self.arena_size - 0.8)
            else:
                # 1D sinusoidal oscillation (original Stages 1-5 behaviour)
                if self.sim_time >= cfg["next_omega_update"]:
                    cfg["omega_target"] = float(rng.uniform(self.omega_min, self.omega_max))
                    cfg["next_omega_update"] = self.sim_time + float(rng.uniform(*self.omega_update_interval))
                cfg["omega"] = ((1.0 - self.omega_smoothing) * cfg["omega"]
                                + self.omega_smoothing * cfg["omega_target"])
                cfg["theta"] += cfg["omega"] * self.dt
                s  = np.sin(cfg["theta"] + cfg["phase"])
                xy = cfg["center"] + cfg["amp"] * s
                xy = np.clip(xy, 0.8, self.arena_size - 0.8)
            p.resetBasePositionAndOrientation(
                obs_id, [float(xy[0]), float(xy[1]), 0.5], [0, 0, 0, 1],
                physicsClientId=self.client)

    def step(self, action):
        lidar_pre = self._get_lidar()
        min_lidar_pre = float(np.min(lidar_pre))
        safe_scale = float(np.clip((min_lidar_pre - 0.45) / 1.15, 0.20, 1.0))
        force_xy = np.clip(action, -1.0, 1.0) * (4.0 * safe_scale)

        for _ in range(self.physics_substeps):
            if self.num_dynamic_obstacles > 0:
                self._update_dynamic_obstacles()
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
            p.applyExternalForce(self.robot, -1, [float(force_xy[0]), float(force_xy[1]), 0.0], robot_pos, p.WORLD_FRAME, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
            self.sim_time += self.dt

        self.current_step += 1
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        robot_xy = np.array(robot_pos[:2], dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(robot_xy - self.goal_pos[:2]))
        step_distance = float(np.linalg.norm(robot_xy - self.prev_xy))
        self.path_length += step_distance

        # Reward balances progress, smoothness and safety.
        progress = self.prev_dist - dist_to_goal
        reward = 12.0 * progress
        reward -= 0.015
        reward -= 0.01 * float(np.sum(np.square(action)))

        lidar = self._get_lidar()
        min_lidar = float(np.min(lidar))
        d_safe = 0.9
        if min_lidar < d_safe:
            ratio = (d_safe - min_lidar) / d_safe
            reward -= 8.0 * (ratio ** 2)

        # Velocity shaping: encourage moving toward the goal, discourage speed near hazards.
        robot_vel = np.array(p.getBaseVelocity(self.robot, physicsClientId=self.client)[0][:2], dtype=np.float32)
        speed = float(np.linalg.norm(robot_vel))
        if speed > 1e-5:
            to_goal = self.goal_pos[:2] - robot_xy
            to_goal_norm = float(np.linalg.norm(to_goal) + 1e-8)
            vel_dir = robot_vel / (speed + 1e-8)
            goal_dir = to_goal / to_goal_norm
            reward += 0.2 * float(np.dot(vel_dir, goal_dir))
        if min_lidar < 1.1:
            reward -= 0.04 * speed * ((1.1 - min_lidar) / 1.1)

        done = False
        truncated = False
        success = False
        collided = False

        contacts = p.getContactPoints(bodyA=self.robot, physicsClientId=self.client)
        hazard_ids = set(self.wall_ids + self.static_ids + self.dynamic_ids)
        if any((c[2] in hazard_ids) for c in contacts):
            reward -= 180.0
            done = True
            collided = True

        if dist_to_goal < 0.55:
            reward += 220.0
            done = True
            success = True

        if self.current_step >= self.max_steps:
            truncated = True

        if step_distance < 0.004:
            reward -= 0.01
        reward += 0.25 * (1.0 - np.tanh(dist_to_goal))

        self.prev_dist = dist_to_goal
        self.prev_xy = robot_xy
        obs = self._get_obs()
        info = {
            "min_lidar": min_lidar,
            "collision": float(collided),
            "success": float(success),
            "path_length": float(self.path_length),
            "dist_to_goal": float(dist_to_goal),
            "episode_length": int(self.current_step),
        }
        return obs, reward, done, truncated, info

    def _get_lidar(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        lidar = []
        n_rays = 24
        for i in range(n_rays):
            angle = 2.0 * np.pi * i / n_rays
            ray_to = [
                robot_pos[0] + np.cos(angle) * self.lidar_range,
                robot_pos[1] + np.sin(angle) * self.lidar_range,
                robot_pos[2],
            ]
            ray = p.rayTest(robot_pos, ray_to, physicsClientId=self.client)
            hit_dist = ray[0][2] * self.lidar_range if ray[0][0] != -1 else self.lidar_range
            lidar.append(hit_dist)
        return np.array(lidar, dtype=np.float32)

    def _get_obs(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        robot_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client)[0][:2]
        lidar = self._get_lidar()
        rel_goal = self.goal_pos[:2] - np.array(robot_pos[:2], dtype=np.float32)
        goal_dist = float(np.linalg.norm(rel_goal) + 1e-8)
        goal_dir = rel_goal / goal_dist
        return np.concatenate([lidar, rel_goal, goal_dir, np.array(robot_vel, dtype=np.float32), [goal_dist]]).astype(np.float32)

    def close(self):
        if hasattr(self, "client"):
            try:
                p.disconnect(self.client)
            except Exception:
                pass


class MixedStageEnv(gym.Env):
    """Samples previous/current stage episodes with configurable probability."""

    def __init__(self, prev_env: gym.Env, curr_env: gym.Env, prev_prob: float = 0.3):
        super().__init__()
        self.prev_env = prev_env
        self.curr_env = curr_env
        self.prev_prob = float(np.clip(prev_prob, 0.0, 1.0))
        self.active_env = self.curr_env
        self.action_space = self.curr_env.action_space
        self.observation_space = self.curr_env.observation_space

    def set_prev_prob(self, value: float):
        self.prev_prob = float(np.clip(value, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        if self.prev_env is None or np.random.rand() >= self.prev_prob:
            self.active_env = self.curr_env
        else:
            self.active_env = self.prev_env
        return self.active_env.reset(seed=seed, options=options)

    def step(self, action):
        return self.active_env.step(action)

    def close(self):
        if self.prev_env is not None:
            self.prev_env.close()
        self.curr_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum SAC training with checkpoint transfer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps_per_stage", type=int, default=1_500_000)
    parser.add_argument("--train_interval_steps", type=int, default=25_000, help="Train this many steps, then run gate evaluation")
    parser.add_argument("--mix_fraction", type=float, default=0.25, help="Fraction of stage steps using 70/30 blend")
    parser.add_argument("--mix_prev_prob", type=float, default=0.30, help="Previous-stage episode probability during blend")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=20_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--stage_start", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--stage_end", type=int, default=5, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--ckpt_freq", type=int, default=100_000)
    parser.add_argument("--gate_eval_episodes", type=int, default=100, help="Episodes for each gate evaluation")
    parser.add_argument("--min_success_rate", type=float, default=0.75, help="Minimum success rate required to pass a stage")
    parser.add_argument("--max_collision_rate", type=float, default=0.15, help="Maximum collision rate allowed to pass a stage")
    parser.add_argument("--stability_window", type=int, default=4, help="Recent eval runs used for stability check")
    parser.add_argument("--success_std_threshold", type=float, default=0.05)
    parser.add_argument("--success_cv_threshold", type=float, default=0.10)
    parser.add_argument("--reward_std_threshold", type=float, default=30.0)
    parser.add_argument("--reward_cv_threshold", type=float, default=0.20)
    parser.add_argument("--enforce_stage_gates", action="store_true", help="If set, move to next stage only when StageGate marks COMPLETE")
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume each requested stage from latest checkpoint in logs/stage_<k>/checkpoints if available",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default="",
        help="Optional explicit checkpoint .zip to resume from (only used for --stage_start stage)",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_stage_config(stage_id: int) -> StageConfig:
    for cfg in DEFAULT_STAGES:
        if cfg.stage_id == stage_id:
            return cfg
    raise ValueError(f"Unknown stage id {stage_id}")


def get_stage_timesteps(stage_id: int) -> Tuple[int, int]:
    if stage_id not in STAGE_TIMESTEPS:
        raise ValueError(f"Unknown stage id {stage_id} for timestep schedule")
    return STAGE_TIMESTEPS[stage_id]


def make_raw_env(stage_cfg: StageConfig, log_path: str, seed: int, render_mode: Optional[str] = None):
    # Stage 6 introduces 2D orbital motion for half the dynamic agents.
    # Stages 1-5 use the default 0.0 (all sinusoidal).
    orbital_fraction = 0.5 if stage_cfg.stage_id == 6 else 0.0
    env = CurriculumObstacleNavEnv(
        num_static_obstacles=stage_cfg.num_static_obstacles,
        num_dynamic_obstacles=stage_cfg.num_dynamic_obstacles,
        arena_size=stage_cfg.arena_size,
        dynamic_speed=stage_cfg.dynamic_speed,
        orbital_fraction=orbital_fraction,
        render_mode=render_mode,
    )
    env = Monitor(env, filename=log_path, info_keywords=("success", "collision", "path_length", "dist_to_goal"))
    env.reset(seed=seed)
    return env


def build_train_env(stage_cfg: StageConfig, stage_log_dir: str, seed: int, prev_stage_cfg: Optional[StageConfig]):
    def _make():
        curr_env = make_raw_env(stage_cfg, os.path.join(stage_log_dir, "train_monitor.csv"), seed=seed)
        if prev_stage_cfg is None:
            return curr_env
        prev_env = make_raw_env(prev_stage_cfg, os.path.join(stage_log_dir, "train_prev_stage_monitor.csv"), seed=seed + 17)
        return MixedStageEnv(prev_env=prev_env, curr_env=curr_env, prev_prob=0.0)

    return DummyVecEnv([_make])


def build_eval_env(stage_cfg: StageConfig, stage_log_dir: str, seed: int):
    def _make():
        return make_raw_env(stage_cfg, os.path.join(stage_log_dir, "eval_monitor.csv"), seed=seed)

    return DummyVecEnv([_make])


def create_new_sac(env, args) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ReLU),
        seed=args.seed,
        verbose=1,
        device=args.device,
    )


def create_transferred_sac(prev_model_path: str, env, args) -> SAC:
    if not os.path.exists(prev_model_path):
        raise FileNotFoundError(f"Previous stage model missing: {prev_model_path}")

    prev_model = SAC.load(prev_model_path, device=args.device)
    new_model = create_new_sac(env=env, args=args)
    new_model.set_parameters(prev_model.get_parameters(), exact_match=False)
    # Fresh model instance => fresh replay buffer for this stage.
    return new_model


def parse_checkpoint_steps(checkpoint_path: str) -> int:
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"_(\d+)_steps\.zip$", filename)
    if match is None:
        return 0
    return int(match.group(1))


def resolve_resume_checkpoint(stage_id: int, ckpt_dir: str, args) -> Optional[str]:
    if stage_id == int(args.stage_start) and args.resume_checkpoint_path:
        if not os.path.exists(args.resume_checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint_path}")
        return args.resume_checkpoint_path

    pattern = os.path.join(ckpt_dir, f"sac_stage_{stage_id}_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=parse_checkpoint_steps)
    return checkpoints[-1]


def set_mix_probability(vec_env: DummyVecEnv, prev_prob: float):
    base_env = vec_env.envs[0]
    if isinstance(base_env, MixedStageEnv):
        base_env.set_prev_prob(prev_prob)


@dataclass
class EvalMetrics:
    success_rate: float
    collision_rate: float
    average_episode_reward: float
    average_episode_length: float
    reward_std: float


class StageGate:
    def __init__(self, stage_id: int, args, csv_path: str):
        self.stage_id = int(stage_id)
        self.min_success_rate = float(args.min_success_rate)
        self.max_collision_rate = float(args.max_collision_rate)
        self.stability_window = int(args.stability_window)
        self.success_std_threshold = float(args.success_std_threshold)
        self.success_cv_threshold = float(args.success_cv_threshold)
        self.reward_std_threshold = float(args.reward_std_threshold)
        self.reward_cv_threshold = float(args.reward_cv_threshold)
        self.history: List[EvalMetrics] = []
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "stage_id,eval_index,train_steps,success_rate,collision_rate,"
                    "reward_mean,reward_std,episode_length_mean,decision\n"
                )

    @staticmethod
    def _cv(values: np.ndarray) -> float:
        mean_abs = float(np.mean(np.abs(values)))
        if mean_abs < 1e-8:
            return 0.0
        return float(np.std(values, ddof=0) / mean_abs)

    def update(self, metrics: EvalMetrics, train_steps: int) -> Dict:
        self.history.append(metrics)
        decision = "advance" if self.is_stage_complete() else "continue"
        row = {
            "stage_id": self.stage_id,
            "eval_index": len(self.history),
            "train_steps": int(train_steps),
            "success_rate": float(metrics.success_rate),
            "collision_rate": float(metrics.collision_rate),
            "reward_mean": float(metrics.average_episode_reward),
            "reward_std": float(metrics.reward_std),
            "episode_length_mean": float(metrics.average_episode_length),
            "decision": decision,
        }
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{row['stage_id']},{row['eval_index']},{row['train_steps']},"
                f"{row['success_rate']:.6f},{row['collision_rate']:.6f},"
                f"{row['reward_mean']:.6f},{row['reward_std']:.6f},"
                f"{row['episode_length_mean']:.6f},{row['decision']}\n"
            )
        return row

    def is_stage_complete(self) -> bool:
        if not self.history:
            return False
        latest = self.history[-1]
        success_ok = latest.success_rate >= self.min_success_rate
        safety_ok = latest.collision_rate <= self.max_collision_rate
        if len(self.history) < self.stability_window:
            return False
        recent = self.history[-self.stability_window :]
        success_vals = np.array([m.success_rate for m in recent], dtype=np.float32)
        reward_vals = np.array([m.average_episode_reward for m in recent], dtype=np.float32)
        success_stable = (
            float(np.std(success_vals, ddof=0)) < self.success_std_threshold
            or self._cv(success_vals) < self.success_cv_threshold
        )
        reward_stable = (
            float(np.std(reward_vals, ddof=0)) < self.reward_std_threshold
            or self._cv(reward_vals) < self.reward_cv_threshold
        )
        return bool(success_ok and safety_ok and success_stable and reward_stable)

    def reset_for_next_stage(self):
        self.history = []

    def load_history_from_csv(self):
        if not os.path.exists(self.csv_path):
            return
        loaded: List[EvalMetrics] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                loaded.append(
                    EvalMetrics(
                        success_rate=float(parts[3]),
                        collision_rate=float(parts[4]),
                        average_episode_reward=float(parts[5]),
                        reward_std=float(parts[6]),
                        average_episode_length=float(parts[7]),
                    )
                )
        if loaded:
            self.history = loaded
            print(f"Loaded {len(self.history)} prior gate evaluations from {self.csv_path}")


def evaluate_policy(model: SAC, stage_cfg: StageConfig, vecnorm_path: str, args) -> EvalMetrics:
    eval_env = build_eval_env(stage_cfg, stage_log_dir=os.path.join(args.log_dir, f"stage_{stage_cfg.stage_id}"), seed=args.seed + 4321)
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    rows: List[Dict] = []
    for _ in range(int(args.gate_eval_episodes)):
        obs = eval_env.reset()
        done = [False]
        ep_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
        ep_info = info[0]
        success = float(ep_info.get("success", 0.0))
        collision = float(ep_info.get("collision", 0.0))
        rows.append(
            {
                "episode_reward": ep_reward,
                "episode_length_steps": int(ep_info.get("episode_length", 0)),
                "success": float(success > 0.5),
                "collision": float(collision > 0.5),
            }
        )
    eval_env.close()
    return EvalMetrics(
        success_rate=float(np.mean([r["success"] for r in rows])),
        collision_rate=float(np.mean([r["collision"] for r in rows])),
        average_episode_reward=float(np.mean([r["episode_reward"] for r in rows])),
        average_episode_length=float(np.mean([r["episode_length_steps"] for r in rows])),
        reward_std=float(np.std([r["episode_reward"] for r in rows], ddof=0)),
    )


def train_step(model: SAC, steps: int, callback: Optional[CheckpointCallback] = None):
    if steps <= 0:
        return
    model.learn(
        total_timesteps=int(steps),
        callback=callback,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False,
    )


def save_stage_artifacts(model: SAC, train_env: VecNormalize, stage_cfg: StageConfig, args) -> Tuple[str, str]:
    model_path_base = os.path.join(args.model_dir, f"sac_stage_{stage_cfg.stage_id}")
    vecnorm_path = os.path.join(args.model_dir, f"vecnormalize_stage_{stage_cfg.stage_id}.pkl")
    model.save(model_path_base)
    train_env.save(vecnorm_path)
    print(f"Saved model: {model_path_base}.zip")
    print(f"Saved vecnorm: {vecnorm_path}")
    return f"{model_path_base}.zip", vecnorm_path


def stage_transition(stage_result: Dict, enforce_stage_gates: bool) -> bool:
    if not enforce_stage_gates:
        return True
    return bool(stage_result.get("completed", False))


def log_stage_training_row(model_dir: str, row: Dict):
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(model_dir, "curriculum_stage_training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "stage_id,min_gate_steps,max_stage_steps,actual_stage_steps,early_terminated,"
                "completed,eval_count,final_success_rate,final_collision_rate,final_reward_mean\n"
            )
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(
            f"{row['stage_id']},{row['min_gate_steps']},{row['max_stage_steps']},"
            f"{row['actual_stage_steps']},{row['early_terminated']},{row['completed']},"
            f"{row['eval_count']},{row['final_success_rate']:.6f},"
            f"{row['final_collision_rate']:.6f},{row['final_reward_mean']:.6f}\n"
        )


def train_one_stage(stage_cfg: StageConfig, prev_stage_cfg: Optional[StageConfig], args) -> Dict:
    stage_dir = os.path.join(args.log_dir, f"stage_{stage_cfg.stage_id}")
    tb_dir = os.path.join(stage_dir, "tb")
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"\n========== Stage {stage_cfg.stage_id} ==========")
    print(
        f"Config: static={stage_cfg.num_static_obstacles}, dynamic={stage_cfg.num_dynamic_obstacles}, "
        f"arena={stage_cfg.arena_size}, dynamic_speed={stage_cfg.dynamic_speed}"
    )

    raw_train_env = build_train_env(stage_cfg, stage_dir, seed=args.seed, prev_stage_cfg=prev_stage_cfg)
    raw_eval_env = build_eval_env(stage_cfg, stage_dir, seed=args.seed + 999)

    stage_vecnorm_path = os.path.join(args.model_dir, f"vecnormalize_stage_{stage_cfg.stage_id}.pkl")
    resume_checkpoint_path: Optional[str] = None
    if args.resume_from_checkpoint:
        resume_checkpoint_path = resolve_resume_checkpoint(stage_cfg.stage_id, ckpt_dir=ckpt_dir, args=args)
        if resume_checkpoint_path:
            print(f"Resuming Stage {stage_cfg.stage_id} from checkpoint: {resume_checkpoint_path}")

    if resume_checkpoint_path and os.path.exists(stage_vecnorm_path):
        train_env = VecNormalize.load(stage_vecnorm_path, raw_train_env)
        print(f"Loaded stage VecNormalize stats: {stage_vecnorm_path}")
    else:
        train_env = VecNormalize(raw_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if resume_checkpoint_path:
            print("Stage VecNormalize stats not found; resuming with fresh normalization statistics.")

    eval_env = VecNormalize(raw_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms

    if resume_checkpoint_path:
        model = SAC.load(resume_checkpoint_path, env=train_env, device=args.device)
    elif stage_cfg.stage_id == 1:
        model = create_new_sac(env=train_env, args=args)
    else:
        prev_model_path = os.path.join(args.model_dir, f"sac_stage_{stage_cfg.stage_id - 1}.zip")
        model = create_transferred_sac(prev_model_path=prev_model_path, env=train_env, args=args)

    model.tensorboard_log = tb_dir

    checkpoint_callback = CheckpointCallback(
        save_freq=args.ckpt_freq,
        save_path=ckpt_dir,
        name_prefix=f"sac_stage_{stage_cfg.stage_id}",
        save_replay_buffer=False,
    )
    min_gate_steps, max_stage_steps = get_stage_timesteps(stage_cfg.stage_id)
    total_steps = int(max_stage_steps)
    interval_steps = int(max(1, args.train_interval_steps))
    mix_steps = int(max(0, min(total_steps, int(total_steps * float(args.mix_fraction)))))
    stage_steps = parse_checkpoint_steps(resume_checkpoint_path) if resume_checkpoint_path else 0
    early_terminated = False
    last_metrics: Optional[EvalMetrics] = None
    gate = StageGate(
        stage_id=stage_cfg.stage_id,
        args=args,
        csv_path=os.path.join(stage_dir, "stage_gate_log.csv"),
    )
    if resume_checkpoint_path:
        gate.load_history_from_csv()
        if stage_steps >= total_steps:
            print(f"Checkpoint already reached stage budget ({stage_steps}/{total_steps}). Running final evaluation only.")

    while stage_steps < total_steps:
        chunk = min(interval_steps, total_steps - stage_steps)
        in_mix_phase = prev_stage_cfg is not None and stage_steps < mix_steps
        set_mix_probability(raw_train_env, prev_prob=float(args.mix_prev_prob if in_mix_phase else 0.0))
        phase_name = "mixed" if in_mix_phase else "current"
        print(f"Stage {stage_cfg.stage_id}: train_step={chunk} ({phase_name} phase), progress={stage_steps}/{total_steps}")
        train_step(model, chunk, callback=checkpoint_callback)
        stage_steps += chunk

        # Do not run gate checks before stage minimum training budget.
        if stage_steps < min_gate_steps:
            model.logger.record("stage_progress/current_stage_steps", float(stage_steps))
            model.logger.record("stage_progress/max_stage_steps", float(total_steps))
            model.logger.record("stage_progress/stage_progress_ratio", float(stage_steps) / float(total_steps))
            model.logger.dump(step=model.num_timesteps)
            continue

        _, vecnorm_path = save_stage_artifacts(model=model, train_env=train_env, stage_cfg=stage_cfg, args=args)
        metrics = evaluate_policy(model=model, stage_cfg=stage_cfg, vecnorm_path=vecnorm_path, args=args)
        last_metrics = metrics
        row = gate.update(metrics=metrics, train_steps=stage_steps)

        model.logger.record("gate/success_rate", metrics.success_rate)
        model.logger.record("gate/collision_rate", metrics.collision_rate)
        model.logger.record("gate/reward_mean", metrics.average_episode_reward)
        model.logger.record("gate/reward_std", metrics.reward_std)
        model.logger.record("gate/episode_length_mean", metrics.average_episode_length)
        model.logger.record("stage_progress/current_stage_steps", float(stage_steps))
        model.logger.record("stage_progress/max_stage_steps", float(total_steps))
        model.logger.record("stage_progress/stage_progress_ratio", float(stage_steps) / float(total_steps))
        model.logger.dump(step=model.num_timesteps)

        print(
            f"[Stage {stage_cfg.stage_id} Eval {row['eval_index']}] "
            f"success={metrics.success_rate:.3f}, collision={metrics.collision_rate:.3f}, "
            f"reward_mean={metrics.average_episode_reward:.2f}, reward_std={metrics.reward_std:.2f}, "
            f"avg_len={metrics.average_episode_length:.1f}, decision={row['decision']}"
        )
        # StageGate has priority over max budget: advance immediately when complete.
        if row["decision"] == "advance":
            print(f"Stage {stage_cfg.stage_id} marked COMPLETE by gate criteria.")
            early_terminated = stage_steps < total_steps
            break

    model_path, vecnorm_path = save_stage_artifacts(model=model, train_env=train_env, stage_cfg=stage_cfg, args=args)
    if last_metrics is None:
        # If no eval happened yet (e.g. very short run), run one final evaluation.
        _, vecnorm_path = save_stage_artifacts(model=model, train_env=train_env, stage_cfg=stage_cfg, args=args)
        last_metrics = evaluate_policy(model=model, stage_cfg=stage_cfg, vecnorm_path=vecnorm_path, args=args)
        gate.update(metrics=last_metrics, train_steps=stage_steps)

    stage_summary = {
        "stage_id": stage_cfg.stage_id,
        "completed": bool(gate.is_stage_complete()),
        "min_gate_steps": int(min_gate_steps),
        "max_stage_steps": int(total_steps),
        "actual_stage_steps": int(stage_steps),
        "early_terminated": bool(early_terminated),
        "eval_count": int(len(gate.history)),
        "final_success_rate": float(last_metrics.success_rate),
        "final_collision_rate": float(last_metrics.collision_rate),
        "final_reward_mean": float(last_metrics.average_episode_reward),
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
    }
    with open(os.path.join(stage_dir, "stage_gate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stage_summary, f, indent=2)
    log_stage_training_row(args.model_dir, stage_summary)

    train_env.close()
    eval_env.close()
    return stage_summary


def save_curriculum_config(args):
    os.makedirs(args.model_dir, exist_ok=True)
    config_path = os.path.join(args.model_dir, "curriculum_config.json")
    payload = {
        "seed": args.seed,
        "timesteps_per_stage": args.timesteps_per_stage,
        "mix_fraction": args.mix_fraction,
        "mix_prev_prob": args.mix_prev_prob,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "train_interval_steps": args.train_interval_steps,
        "gate_eval_episodes": args.gate_eval_episodes,
        "min_success_rate": args.min_success_rate,
        "max_collision_rate": args.max_collision_rate,
        "stability_window": args.stability_window,
        "success_std_threshold": args.success_std_threshold,
        "success_cv_threshold": args.success_cv_threshold,
        "reward_std_threshold": args.reward_std_threshold,
        "reward_cv_threshold": args.reward_cv_threshold,
        "enforce_stage_gates": bool(args.enforce_stage_gates),
        "stage_timesteps": {str(k): list(v) for k, v in STAGE_TIMESTEPS.items()},
        "stages": [cfg.__dict__ for cfg in DEFAULT_STAGES],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved curriculum config: {config_path}")


def main():
    args = parse_args()
    if args.stage_start > args.stage_end:
        raise ValueError("--stage_start must be <= --stage_end")

    save_curriculum_config(args)

    stage_results: List[Dict] = []
    for stage_id in range(args.stage_start, args.stage_end + 1):
        stage_cfg = get_stage_config(stage_id)
        prev_cfg = get_stage_config(stage_id - 1) if stage_id > 1 else None
        stage_result = train_one_stage(stage_cfg=stage_cfg, prev_stage_cfg=prev_cfg, args=args)
        stage_results.append(stage_result)
        if not stage_transition(stage_result, args.enforce_stage_gates):
            print(
                f"\nStopping curriculum: Stage {stage_id} not COMPLETE under gate criteria. "
                "Tune settings or increase timesteps, then resume from this stage."
            )
            break

    summary_path = os.path.join(args.model_dir, "curriculum_stage_transition_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stage_results, f, indent=2)
    print(f"Saved transition summary: {summary_path}")

    print("\nCurriculum training completed.")


if __name__ == "__main__":
    main()
