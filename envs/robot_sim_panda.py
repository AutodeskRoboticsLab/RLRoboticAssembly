import math
import pybullet as p
import numpy as np
import utilities as util

INITIAL_POS = [0.0, 0.0, 0.0]
INITIAL_ORN = [0, 0, 0, 1]
TARGET_POS = [0.6, 0.044, 0]
TARGET_ORN = [0, 0, 0, 1]
HOLE_OFFSET = [0.048485, -0.04305, -0.03]

URDF_PATH_TOOL = 'envs/urdf/panda_peg_in_hole/panda_peg'
URDF_PATH_TARGET = 'envs/urdf/panda_peg_in_hole/insertion_box'

# Official initial positions for Panda [0, -45, 0, -135, 0, 90, 45]
INITIAL_PANDA_JOINTS = [math.radians(0), math.radians(20), math.radians(0), math.radians(-103), math.radians(0),
                        math.radians(122), math.radians(45)]


class RobotSimPanda():

    def __init__(self):

        self.uid = p.loadURDF(util.format_urdf_filepath(URDF_PATH_TOOL),
                              basePosition=INITIAL_POS,
                              baseOrientation=INITIAL_ORN,
                              useFixedBase=1)

        self.target_uid = p.loadURDF(util.format_urdf_filepath(URDF_PATH_TARGET),
                                     basePosition=TARGET_POS,
                                     baseOrientation=TARGET_ORN,
                                     useFixedBase=1)
        self.link_target = 0
        util.display_frame_axis(self.target_uid, self.link_target)

        # In urdf, the panda robot has 11 joints
        # 7 joints belong to the arm, each with a FT sensor
        self.num_arm_joints = 7

        # FT sensor at the last arm joint
        self.link_sensor = 6
        util.display_frame_axis(self.uid, self.link_sensor)

        # The 8th link: a virtual link at the end of the arm, as the end effector
        self.link_ee = 7
        util.display_frame_axis(self.uid, self.link_ee)

        self.link_member = 8
        util.display_frame_axis(self.uid, self.link_member)

        self.max_force = 200
        self.max_velocity = 0.35

        self.joint_positions = INITIAL_PANDA_JOINTS

        # Apply the joint positions
        for jointIndex in range(self.num_arm_joints):
            p.resetJointState(self.uid, jointIndex, self.joint_positions[jointIndex])
            p.setJointMotorControl2(self.uid, jointIndex, p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions[jointIndex], force=self.max_force,
                                    maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)

        # Important to call getLinkState only once at the beginning, otherwise PyBullet bug makes the robot unstable.
        ee_pose = p.getLinkState(self.uid, self.link_ee)
        self.ee_position = list(ee_pose[0])
        self.ee_orientation = list(ee_pose[1])

    def get_member_pose(self):
        link_member_pose = p.getLinkState(self.uid, self.link_member)
        return [link_member_pose[0], link_member_pose[1]]

    def get_target_pose(self):
        pose = p.getLinkState(self.target_uid, self.link_target)[4:6]
        # Offset the hole center from corner
        return [(np.array(pose[0]) + HOLE_OFFSET).tolist(), pose[1]]

    def enable_force_torque_sensor(self):
        # FT sensor is measured at the center of mass in Pybullet
        p.enableJointForceTorqueSensor(self.uid, self.link_sensor)

    def get_force_torque(self):
        # FT reading in pybullet is negated and needs to be scaled down
        ft = np.multiply(-0.1, p.getJointState(self.uid, self.link_sensor)[2]).tolist()
        return ft

    def apply_action_pose(self, delta_pose):
        for i in range(3):
            self.ee_position[i] += delta_pose[i]

        orn = list(p.getEulerFromQuaternion(self.ee_orientation))
        for i in range(3):
            orn[i] += delta_pose[i+3]
        self.ee_orientation = p.getQuaternionFromEuler(orn)

        joint_positions = p.calculateInverseKinematics(self.uid, self.num_arm_joints, self.ee_position, self.ee_orientation)

        for i in range(self.num_arm_joints):
            p.setJointMotorControl2(bodyUniqueId=self.uid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_positions[i], targetVelocity=0, force=self.max_force,
                                    maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)

    def apply_action_position(self, delta_pos):
        for i in range(3):
            self.ee_position[i] += delta_pos[i]

        joint_positions = p.calculateInverseKinematics(self.uid, self.num_arm_joints, self.ee_position, self.ee_orientation)

        for i in range(self.num_arm_joints):
            p.setJointMotorControl2(bodyUniqueId=self.uid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_positions[i], targetVelocity=0, force=self.max_force,
                                    maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)


