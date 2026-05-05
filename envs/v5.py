import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np

class SafetySphereNavEnv(gym.Env):
    def __init__(self, render_mode=None, curriculum_progress=1.0, arena_size=6.0):
        super().__init__()
        self.render_mode = render_mode
        self.curriculum_progress = float(np.clip(curriculum_progress, 0.0, 1.0))
        self.arena_size = float(arena_size)
        self.physics_substeps = 6

        # Action: vx, vy (clipped to reasonable values)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 24 lidar + relative goal pos(2) + goal_dir(2) + vel(2) + dist(1)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        self.max_steps = int(np.clip(1000.0 * (self.arena_size / 4.8), 1000, 1500))
        self.current_step = 0
        far_xy = self.arena_size - 1.0
        self.goal_pos = np.array([far_xy, far_xy, 0.5], dtype=np.float32)

    def set_curriculum_progress(self, value):
        self.curriculum_progress = float(np.clip(value, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            p.disconnect()
        except:
            pass

        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.resetDebugVisualizerCamera(
            cameraDistance=7, cameraYaw=45, cameraPitch=-60,
            cameraTargetPosition=[2.5, 2.5, 0]
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 240.0)

        # Ground
        self.plane_id = p.loadURDF("plane.urdf")

        # Walls (slightly inset for safety margin)
        wall_thickness = 0.2
        wall_height = 2
        arena_size = self.arena_size

        walls = [
            ([-0.1, arena_size / 2, wall_height / 2], [wall_thickness, arena_size, wall_height]),
            ([arena_size + 0.1, arena_size / 2, wall_height / 2], [wall_thickness, arena_size, wall_height]),
            ([arena_size / 2, -0.1, wall_height / 2], [arena_size, wall_thickness, wall_height]),
            ([arena_size / 2, arena_size + 0.1, wall_height / 2], [arena_size, wall_thickness, wall_height])
        ]

        self.wall_ids = []
        for pos, size in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=[0.7, 0.7, 0.7, 1])
            wall_id = p.createMultiBody(0, col, vis, basePosition=pos)
            self.wall_ids.append(wall_id)

        # Robot (smaller for better maneuverability)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0, 0, 1, 1])

        # Curriculum:
        # - early training: shorter horizon and more central starts
        # - later training: full goal and wider start distribution
        near_xy = 0.55 * arena_size
        far_xy = arena_size - 1.0
        near_goal = np.array([near_xy, near_xy, 0.5], dtype=np.float32)
        far_goal = np.array([far_xy, far_xy, 0.5], dtype=np.float32)
        self.goal_pos = (1.0 - self.curriculum_progress) * near_goal + self.curriculum_progress * far_goal

        center_xy = 0.35 * arena_size
        early_half_width = 0.10 * arena_size
        late_half_width = 0.18 * arena_size
        half_width = (1.0 - self.curriculum_progress) * early_half_width + self.curriculum_progress * late_half_width
        start_min = max(0.6, center_xy - half_width)
        start_max = min(arena_size - 0.6, center_xy + half_width)
        start_x = np.random.uniform(start_min, start_max)
        start_y = np.random.uniform(start_min, start_max)
        # Keep the sphere slightly above the plane so it settles cleanly
        start_pos = [start_x, start_y, 0.26]

        self.robot = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=start_pos
        )
        p.changeDynamics(self.robot, -1, linearDamping=0.05, angularDamping=0.05, lateralFriction=0.8)

        # Goal
        goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(0, baseVisualShapeIndex=goal_vis, basePosition=self.goal_pos)

        self.prev_dist = np.linalg.norm(np.array(start_pos[:2]) - self.goal_pos[:2])
        self.prev_xy = np.array(start_pos[:2], dtype=np.float32)
        self.path_length = 0.0
        self.current_step = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Slightly reduced control aggressiveness for smoother policy learning
        vx, vy = np.clip(action, -1.0, 1.0) * 4.0

        # Apply force at robot center across multiple physics steps so each RL
        # action produces clearer motion and learning signal.
        for _ in range(self.physics_substeps):
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
            p.applyExternalForce(self.robot, -1, [vx, vy, 0], robot_pos, p.WORLD_FRAME)
            p.stepSimulation()

        self.current_step += 1
        obs = self._get_obs()

        self.collided_this_step = False

        # State extraction
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_vel = p.getBaseVelocity(self.robot)[0][:2]
        robot_xy = np.array(robot_pos[:2], dtype=np.float32)
        dist_to_goal = np.linalg.norm(robot_xy - self.goal_pos[:2])
        step_distance = float(np.linalg.norm(robot_xy - self.prev_xy))
        self.path_length += step_distance

        # ---------------- REWARD DESIGN ---------------- #
        reward = 0.0

        # 1) Progress reward: positive when moving toward goal, negative otherwise
        progress = self.prev_dist - dist_to_goal
        reward += 8.0 * progress

        # 2) Per-step time penalty to avoid dithering
        reward -= 0.02

        # 3) Small control effort penalty for smoother actions
        reward -= 0.01 * float(np.sum(np.square(action)))

        done = False
        truncated = False

        # 4) Safety penalty near obstacles/walls (potential-shaped)
        lidar = self._get_lidar()
        min_lidar = np.min(lidar)
        d_safe = 0.55
        if min_lidar < d_safe:
            safety_ratio = (d_safe - min_lidar) / d_safe
            reward -= 4.0 * (safety_ratio ** 2)

        # 5) Collision event (terminate on crash)
        # Ignore expected plane contact; only count collisions with walls.
        contacts = p.getContactPoints(bodyA=self.robot)
        collided_with_wall = any((c[2] in getattr(self, "wall_ids", [])) for c in contacts)
        if collided_with_wall:
            reward -= 120.0
            self.collided_this_step = True
            done = True

        # 5b) Small distance-shaping term to keep long-range pull to the goal.
        reward += 0.3 * (1.0 - np.tanh(dist_to_goal))

        self.prev_dist = dist_to_goal

        # ---------------- TERMINATION ---------------- #
        success = False
        if dist_to_goal < 0.45:
            reward += 150.0
            done = True
            success = True

        if self.current_step >= self.max_steps:
            truncated = True

        self.prev_xy = robot_xy

        info = {"min_lidar": min_lidar,
                "collision": float(self.collided_this_step),
                "success": float(success),
                "path_length": float(self.path_length),
                "dist_to_goal": dist_to_goal,
                "episode_length": self.current_step}

        return obs, reward, done, truncated, info

    def _get_lidar(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        lidar = []

        for i in range(24):
            angle = 2 * np.pi * i / 24
            ray_to = [
                robot_pos[0] + np.cos(angle) * 6.0,
                robot_pos[1] + np.sin(angle) * 6.0,
                robot_pos[2]
            ]
            ray = p.rayTest(robot_pos, ray_to)
            hit_dist = ray[0][2] if ray[0][0] != -1 else 6.0
            lidar.append(hit_dist)

        return np.array(lidar, dtype=np.float32)

    def _get_obs(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_vel = p.getBaseVelocity(self.robot)[0][:2]
        lidar = self._get_lidar()

        # RELATIVE goal position (CRITICAL for DDPG)
        rel_goal_pos = self.goal_pos[:2] - np.array(robot_pos[:2])
        goal_dist = np.linalg.norm(rel_goal_pos) + 1e-8
        goal_dir = rel_goal_pos / goal_dist

        obs = np.concatenate([
            lidar,
            rel_goal_pos,  # Relative position instead of absolute
            goal_dir,
            robot_vel,
            [goal_dist]    # Distance scalar
        ]).astype(np.float32)

        return obs

    def close(self):
        p.disconnect()