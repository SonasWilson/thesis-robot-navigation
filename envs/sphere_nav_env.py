import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import time


class SphereNavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Actions: move in x and y
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observations: 24 lidar + x,y position + vx,vy velocity = 28
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        self.max_steps = 500
        self.current_step = 0


    def reset(self, seed=None, options=None):
        try:
            p.disconnect()
        except:
            pass

        self.client = p.connect(p.DIRECT)

        # Better camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=7,
            cameraYaw=45,
            cameraPitch=-60,
            cameraTargetPosition=[2.5, 2.5, 0]
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Ground
        p.loadURDF("plane.urdf")

        # Create BIG walls using boxes
        wall_thickness = 0.2
        wall_height = 2
        arena_size = 5

        walls = [
            ([0, arena_size / 2, wall_height / 2], [wall_thickness, arena_size, wall_height]),  # left
            ([arena_size, arena_size / 2, wall_height / 2], [wall_thickness, arena_size, wall_height]),  # right
            ([arena_size / 2, 0, wall_height / 2], [arena_size, wall_thickness, wall_height]),  # bottom
            ([arena_size / 2, arena_size, wall_height / 2], [arena_size, wall_thickness, wall_height])  # top
        ]

        for pos, size in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=[0.7, 0.7, 0.7, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

        # Robot (BLUE sphere)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0,0,1,1])

        self.robot = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[1,1,0.5]
        )

        # Goal (RED sphere)
        self.goal_pos = np.array([4,4,0.5])

        goal_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.4,
            rgbaColor=[1,0,0,1]
        )

        self.goal_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_vis,
            basePosition=self.goal_pos
        )

        self.current_step = 0

        obs = self._get_obs()
        return obs, {}


    def step(self, action):
        vx, vy = action

        # Apply movement force
        p.applyExternalForce(self.robot, -1, [vx*20, vy*20, 0], [0,0,0], p.WORLD_FRAME)
        p.stepSimulation()

        self.current_step += 1

        obs = self._get_obs()

        # Get robot position
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        dist_to_goal = np.linalg.norm(np.array(robot_pos[:2]) - self.goal_pos[:2])

        # Reward
        reward = -dist_to_goal * 0.1

        done = False
        truncated = self.current_step >= self.max_steps

        # Collision or success
        contacts = p.getContactPoints(self.robot)

        if contacts or dist_to_goal < 0.3:
            if dist_to_goal < 0.3:
                reward += 100
            else:
                reward -= 100
            done = True

        # Slow down so you can SEE
        time.sleep(1./60)

        return obs, reward, done, truncated, {}


    def _get_obs(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_vel = p.getBaseVelocity(self.robot)[0][:2]

        lidar = []
        for i in range(24):
            angle = 2 * np.pi * i / 24
            ray_to = [
                robot_pos[0] + np.cos(angle) * 5,
                robot_pos[1] + np.sin(angle) * 5,
                robot_pos[2]
            ]

            ray = p.rayTest(robot_pos, ray_to)
            hit_dist = ray[0][2] if ray[0][0] != -1 else 5.0
            lidar.append(hit_dist)

        obs = np.array(lidar + list(robot_pos[:2]) + list(robot_vel), dtype=np.float32)
        return obs


    def close(self):
        p.disconnect()