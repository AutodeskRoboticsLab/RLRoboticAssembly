from envs.robot_sim_robotless import RobotSimRobotless
from envs.robot_sim_panda import RobotSimPanda
from envs.robot_real_example import RobotRealExample

t = 'sim'  # sim or real?

if t == 'sim':
    from envs.task_sim import TaskSim
    def env_creator(env_config):
        environment = TaskSim(env_robot=RobotSimRobotless,  # choose the sim robot class
                              self_collision_enabled=True,  # collision setting for pybullet
                              renders=True,  # normally for running sim and rolling out in sim, this is set to True; for training, False.
                              ft_noise=False,  # domain randomization on force/torque observation
                              pose_noise=False,  # domain randomization on pose observation
                              action_noise=False,  # domain randomization on actions
                              physical_noise=False,  # domain randomization on physical parameters
                              time_step=1/250,  # sets the control frequency of the robot
                              max_steps=4000,  # max number of steps in each episode
                              step_limit=True,  # limit the length of an episode by max_steps?
                              action_dim=6,  # dimension of action space
                              max_vel=0.01,  # max linear velocity (m/s) along each axis
                              max_rad=0.01,  # max rotational velocity (rad/s) around each axis
                              ft_obs_only=False,  # only use force/torque as observation?
                              limit_ft=False,  # limit force/torque based on max_ft?
                              max_ft=[1000, 1000, 2500, 100, 100, 100],  # max force (N) and torque (Nm)
                              max_position_range=[2]*3,  # max observation space for positions (m)
                              dist_threshold=0.005)  # an episode is considered successful when distance is within the threshold.

        return environment


if t == 'real':
    from envs.task_real import TaskReal
    def env_creator(env_config):
        environment = TaskReal(env_robot=RobotRealExample,  # choose the real robot class
                               time_step=1/250,
                               max_steps=4000,
                               step_limit=True,
                               action_dim=6,
                               max_vel=0.01,
                               max_rad=0.01,
                               ft_obs_only=False,
                               limit_ft=False,
                               max_ft=[667.233, 667.233, 2001.69, 67.7908, 67.7908, 67.7908],  # ATI-Delta FT sensor limits
                               max_position_range=[2] * 3,
                               dist_threshold=0.005)

        return environment


