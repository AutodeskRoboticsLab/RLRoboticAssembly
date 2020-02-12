# Task setup for working with the real robot and sensor

import numpy as np

from envs.task import Task


class TaskReal(Task):

    def __init__(self,
                 env_robot=None,
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

        self.env = env_robot()

    def reset(self):
        self.max_dist = self.dist_to_target()
        self._env_step_counter = 0
        self._observation = self.get_extended_observation()

        return np.array(self._observation)

    def get_member_pose(self):
        return self.env.get_member_pose()

    def get_target_pose(self):
        return self.env.get_target_pose()

    def get_force_torque(self):
        return self.env.get_force_torque()

    def step2(self, delta):
        reward, done, num_success = self.reward()

        if done:
            if self.action_dim > 3:
                last_delta = [0.0] * 6
                self.env.apply_action_pose(last_delta, 1)
            else:
                last_delta = [0.0] * 3
                self.env.apply_action_position(last_delta, 1)
        else:
            if self.action_dim > 3:
                self.env.apply_action_pose(delta, 0)
            else:
                self.env.apply_action_position(delta, 0)

        self._env_step_counter += 1

        self._observation = self.get_extended_observation()

        return np.array(self._observation), reward, done, {"num_success": num_success}
