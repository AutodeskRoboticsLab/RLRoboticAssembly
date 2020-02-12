import utilities as util

# modules for interfacing with robot control and FT sensor
# from robot_control_interface import ControlInterface
# from ft_sensor_interface import FTInterface


class RobotRealExample():

    def __init__(self):

        # Connect the interfaces
        self.robot_interface = ControlInterface()
        self.ft_interface = FTInterface()

    @staticmethod
    def decompose_incoming_pose_data(data):
        position = data[:3]
        rotation = data[3:7]

        return [position, rotation]

    def get_member_pose(self):
        self.robot_interface.receive()
        data_in = self.robot_interface.message_in.values
        values = self.decompose_incoming_pose_data(data_in)
        position_m = values[0]
        rotation_quat = values[1]

        return [position_m, rotation_quat]

    @staticmethod
    def get_target_pose():
        # target at world origin
        return [0, 0, 0], util.xyzw_by_euler([0, 0, 0], 'sxyz')

    def get_force_torque(self):
        self.ft_interface.receive()
        data_in = self.ft_interface.message_in.values
        force_torque = data_in

        return force_torque

    def apply_action_pose(self, delta, done):
        relative_pos = delta[0:3]
        relative_orn = delta[3:6]

        data_out = list(relative_pos) + list(relative_orn) + [done]

        self.robot_interface.send(data_out)

    def apply_action_position(self, delta, done):
        data_out = list(delta) + [0, 0, 0] + [done]

        self.robot_interface.send(data_out)
