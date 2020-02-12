import pybullet as p
import time

import numpy as np
from gym.utils import seeding

from envs.task import Task


class TaskSim(Task):

    def __init__(self,
                 env_robot=None,
                 self_collision_enabled=None,
                 renders=None,
                 ft_noise=None,
                 pose_noise=None,
                 action_noise=None,
                 physical_noise=None,
                 time_step=None,
                 max_steps=None,
                 step_limit=None,
                 action_dim=None,
                 max_vel=None,
                 max_rad=None,
                 ft_obs_only=None,
                 limit_ft=None,
                 max_ft=None,
                 max_position_range=None,
                 dist_threshold=None):

        super().__init__(max_steps=max_steps,
                         action_dim=action_dim,
                         step_limit=step_limit,
                         max_vel=max_vel,
                         max_rad=max_rad,
                         ft_obs_only=ft_obs_only,
                         limit_ft=limit_ft,
                         time_step=time_step,
                         max_ft=max_ft,
                         max_position_range=max_position_range,
                         dist_threshold=dist_threshold)

        self._env_robot = env_robot
        self._self_collision_enabled = self_collision_enabled
        self._renders = renders

        """ parameters to control the level of domain randomization """
        self._ft_noise = ft_noise
        self._ft_noise_level = [0.5, 0.5, 0.5, 0.05, 0.05, 0.05]  # N N N Nm Nm Nm
        self._ft_bias_level = [2.0, 2.0, 2.0, 0.2, 0.2, 0.2]  # N N N Nm Nm Nm
        self._ft_bias = 0.0
        self._pose_noise = pose_noise
        self._pos_noise_level = 0.001  # m
        self._orn_noise_level = 0.001  # rad
        self._action_noise = action_noise
        self._action_noise_lin = 0.001  # multiplier for linear translation
        self._action_noise_rot = 0.001  # multiplier for rotation
        self._physical_noise = physical_noise
        self._friction_noise_level = 0.1

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
        else:
            p.connect(p.DIRECT)

        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.seed()

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
        p.setTimeStep(self._time_step)
        p.setGravity(0, 0, 0)

        self.env = self._env_robot()
        self.max_dist = self.dist_to_target()

        self._env_step_counter = 0
        self.env.enable_force_torque_sensor()
        p.stepSimulation()

        self.correlated_noise()

        self._observation = self.get_extended_observation()
        self.add_observation_noise()

        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_member_pose(self):
        return self.env.get_member_pose()

    def get_target_pose(self):
        return self.env.get_target_pose()

    def get_force_torque(self):
        return self.env.get_force_torque()

    def step2(self, delta):
        reward, done, num_success = self.reward()

        if not done:
            if self.action_dim > 3:
                self.env.apply_action_pose(delta)
            else:
                self.env.apply_action_position(delta)
        else:
            if self.action_dim > 3:
                self.env.apply_action_pose([0.0] * 6)
            else:
                self.env.apply_action_position([0.0] * 3)

        p.stepSimulation()
        self._env_step_counter += 1

        if self._renders:
            time.sleep(self._time_step)

        self._observation = self.get_extended_observation()
        self.add_observation_noise()

        return np.array(self._observation), reward, done, {"num_success": num_success}

    def correlated_noise(self):
        # force torque sensor bias, sampled at the start of each episode
        if self._ft_noise:
            self._ft_bias = self.add_gaussian_noise(0.0, self._ft_bias_level, [0] * len(self._ft_bias_level))

        # add some physical noise here
        if self._physical_noise:
            self.add_all_friction_noise(self._friction_noise_level)

    def uncorrelated_pose_noise(self, pos, orn):
        if self._pose_noise:
            pos = self.add_gaussian_noise(0.0, self._pos_noise_level, pos)
            orn_euler = p.getEulerFromQuaternion(orn)
            orn_euler = self.add_gaussian_noise(0.0, self._orn_noise_level, orn_euler)
            orn = p.getQuaternionFromEuler(orn_euler)

        return pos, orn

    def uncorrelated_position_noise(self, pos):
        if self._pose_noise:
            pos = self.add_gaussian_noise(0.0, self._pos_noise_level, pos)

        return pos

    def add_ft_noise(self, force_torque):
        if self._ft_noise:
            force_torque = self.add_gaussian_noise(0.0, self._ft_noise_level, force_torque)
            force_torque = np.add(force_torque, self._ft_bias).tolist()

        return force_torque

    def add_observation_noise(self):
        if not self._ft_obs_only:
            if self.action_dim > 3:
                self._observation[0:3], self._observation[3:7] = self.uncorrelated_pose_noise(self._observation[0:3],
                                                                                              self._observation[3:7])
                self._observation[7:13] = self.add_ft_noise(self._observation[7:13])
            else:
                self._observation[0:3] = self.uncorrelated_position_noise(self._observation[0:3])
                self._observation[3:9] = self.add_ft_noise(self._observation[3:9])
        else:
            self._observation[0:6] = self.add_ft_noise(self._observation[0:6])

    def add_all_friction_noise(self, noise_level):
        self.add_body_friction_noise(self.env.target_uid, self.env.link_target, noise_level)
        self.add_body_friction_noise(self.env.uid, self.env.link_member, noise_level)

    @staticmethod
    def add_body_friction_noise(uid, link, noise_level):
        dynamics = p.getDynamicsInfo(uid, link)
        noise_range = np.fabs(dynamics[1]) * noise_level
        friction_noise = np.random.normal(0, noise_range)
        p.changeDynamics(uid, link, lateralFriction=dynamics[1] + friction_noise)

    @staticmethod
    def add_gaussian_noise(mean, std, vec):
        noise = np.random.normal(mean, std, np.shape(vec))
        return np.add(vec, noise).tolist()
