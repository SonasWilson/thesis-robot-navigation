import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time


class StaticObstacleNavEnv(gym.Env):
    def __init__(self, render_mode=None, arena_size=10.0):
        super().__init__()
        self.render_mode = render_mode
        self.arena_size = float(arena_size)
        self.max_steps = 1800
        self.physics_substeps = 6
        self.lidar_range = 10.0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        # Fixed goal for all episodes in this stage.
        self.goal_pos = np.array([7.8, 7.8, 0.5], dtype=np.float32)
        self.obstacle_specs = [
            {"pos": [4.0, 5.5, 0.5], "half_extents": [0.6, 1.0, 0.5]},
            {"pos": [6.8, 3.6, 0.5], "half_extents": [0.8, 0.5, 0.5]},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            p.disconnect()
        except Exception:
            pass

        self.client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=12.0,
                cameraYaw=0.0,
                cameraPitch=-89.0,
                cameraTargetPosition=[self.arena_size / 2.0, self.arena_size / 2.0, 0.0],
            )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 240.0)

        p.loadURDF("plane.urdf")
        self.wall_ids = self._create_walls()
        self.obstacle_ids = self._create_static_obstacles()
        self._create_goal_visual()
        self.robot = self._create_robot()

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        self.prev_xy = np.array(robot_pos[:2], dtype=np.float32)
        self.prev_dist = float(np.linalg.norm(self.prev_xy - self.goal_pos[:2]))
        self.path_length = 0.0
        self.current_step = 0

        return self._get_obs(), {}

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
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size])
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=[0.7, 0.7, 0.7, 1.0]
            )
            wall_ids.append(p.createMultiBody(0, col, vis, basePosition=pos))
        return wall_ids

    def _create_static_obstacles(self):
        ids = []
        for spec in self.obstacle_specs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=spec["half_extents"])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=spec["half_extents"], rgbaColor=[0.9, 0.6, 0.1, 1.0])
            ids.append(p.createMultiBody(0, col, vis, basePosition=spec["pos"]))
        return ids

    def _create_goal_visual(self):
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.35, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(0, baseVisualShapeIndex=goal_vis, basePosition=self.goal_pos)

    def _create_robot(self):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0, 0, 1, 1])
        start_pos = [1.2, 1.2, 0.26]
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=start_pos,
        )
        p.changeDynamics(robot_id, -1, linearDamping=0.05, angularDamping=0.05, lateralFriction=0.8)
        return robot_id

    def step(self, action):
        lidar_pre = self._get_lidar()
        min_lidar_pre = float(np.min(lidar_pre))
        safe_scale = float(np.clip((min_lidar_pre - 0.45) / 1.1, 0.25, 1.0))
        force_xy = np.clip(action, -1.0, 1.0) * (4.0 * safe_scale)
        for _ in range(self.physics_substeps):
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
            p.applyExternalForce(self.robot, -1, [float(force_xy[0]), float(force_xy[1]), 0], robot_pos, p.WORLD_FRAME)
            p.stepSimulation()

        self.current_step += 1
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(robot_pos[:2], dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(robot_xy - self.goal_pos[:2]))
        step_distance = float(np.linalg.norm(robot_xy - self.prev_xy))
        self.path_length += step_distance

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

        robot_vel = np.array(p.getBaseVelocity(self.robot)[0][:2], dtype=np.float32)
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

        contacts = p.getContactPoints(bodyA=self.robot)
        hazard_ids = set(self.wall_ids + self.obstacle_ids)
        if any((c[2] in hazard_ids) for c in contacts):
            reward -= 180.0
            done = True
            collided = True

        if dist_to_goal < 0.55:
            reward += 200.0
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
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        lidar = []
        for i in range(24):
            angle = 2.0 * np.pi * i / 24.0
            ray_to = [
                robot_pos[0] + np.cos(angle) * self.lidar_range,
                robot_pos[1] + np.sin(angle) * self.lidar_range,
                robot_pos[2],
            ]
            ray = p.rayTest(robot_pos, ray_to)
            hit_dist = ray[0][2] * self.lidar_range if ray[0][0] != -1 else self.lidar_range
            lidar.append(hit_dist)
        return np.array(lidar, dtype=np.float32)

    def _get_obs(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_vel = p.getBaseVelocity(self.robot)[0][:2]
        lidar = self._get_lidar()

        rel_goal = self.goal_pos[:2] - np.array(robot_pos[:2], dtype=np.float32)
        goal_dist = float(np.linalg.norm(rel_goal) + 1e-8)
        goal_dir = rel_goal / goal_dist

        return np.concatenate(
            [lidar, rel_goal, goal_dir, np.array(robot_vel, dtype=np.float32), [goal_dist]]
        ).astype(np.float32)

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass


def preview_env(episodes=3):
    env = StaticObstacleNavEnv(render_mode="human")
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            while not (done or truncated):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                time.sleep(1.0 / 60.0)
            print(
                f"[Static2] Episode {ep + 1}: reward={ep_reward:.2f}, "
                f"success={info.get('success', 0.0):.0f}, "
                f"collision={info.get('collision', 0.0):.0f}, "
                f"path={info.get('path_length', 0.0):.2f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    preview_env(episodes=3)
