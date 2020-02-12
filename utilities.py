import csv
import numpy
import os
import inspect
import transforms3d
import pybullet as p

# print in color for better readability
def prGreen(skk):
    print("\033[92m {}\033[00m" .format(skk))


def prRed(skk):
    print("\033[91m {}\033[00m" .format(skk))


def display_frame_axis(body_uid, link_index, line_length=0.05):
    # Red: X axis, Green: Y axis, Blue: Z axis

    p.addUserDebugLine([0, 0, 0], [line_length, 0, 0], [1, 0, 0],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
    p.addUserDebugLine([0, 0, 0], [0, line_length, 0], [0, 1, 0],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
    p.addUserDebugLine([0, 0, 0], [0, 0, line_length], [0, 0, 1],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)


def write_csv(data, csv_file, overwrite):
    if os.path.isfile(csv_file) & overwrite:
        os.remove(csv_file)
    with open(csv_file, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


def format_urdf_filepath(name):
    dot_urdf = '.urdf'
    if dot_urdf not in name:
        name += dot_urdf
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # print("current_dir=" + currentdir)
    os.sys.path.insert(0, currentdir)

    return '{}/{}'.format(currentdir, name)


def qinverse(q):
    return transforms3d.quaternions.qinverse(q[3], q[0], q[1], q[2])


def xyzw_by_euler(euler, axes):
    q = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2], axes)  # s for static = extrinsic
    return wxyz_to_xyzw(q)


def quat_to_euler(xyzw, axes):  # axes specifies the type of Euler wanted
    return transforms3d.euler.quat2euler(xyzw_to_wxyz(xyzw), axes)


def xyzw_to_wxyz(xyzw):
    wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return wxyz


def wxyz_to_xyzw(wxyz):
    xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    return xyzw

# Real robot only
def mat33_by_abc(abc):
    """
    Get matrix from Euler Angles
    :param abc:
    :return:
    """
    a, b, c = abc
    return transforms3d.euler.euler2mat(a, b, c, axes='rzyx')  # r for rotating = intrinsic


def mat33_by_mat44(mat):
    """
    Extract a 3x3 matrix from a 4x4 matrix.
    :param mat:
    :return:
    """
    m = numpy.eye(3, 3)
    for i in range(3):
        for j in range(3):
            m[i][j] = mat[i][j]
    return m


def mat33_by_quat(xyzw):
    """
    Convert quaternion to matrix.
    :param xyzw: list, quaternion [x, y, z, w]
    :return: 3x3 matrix
    """
    wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return transforms3d.quaternions.quat2mat(wxyz)


def mat33_to_quat(mat):
    """
    Convert matrix to quaternion.
    :param mat: 3x3 matrix
    :return: list, quaternion [x, y, z, w]
    """
    wxyz = transforms3d.quaternions.mat2quat(mat)
    return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


# Real robot only
def mat33_to_abc(mat):
    """
    Get Euler Angles from matrix.
    :param mat: 3x3 matrix
    :return: list, Euler Angles [z, y', x"]
    """
    return list(transforms3d.euler.mat2euler(mat, axes='rzyx'))


def mat44_by_pos_mat33(pos, mat):
    """
    Construct a matrix using a position and matrix.
    :param pos: list, position [x, y, z]
    :param mat: 3x3 matrix.
    :return: 4x4 matrix
    """
    m = numpy.eye(4, 4)
    for i in range(3):
        for j in range(3):
            m[i][j] = mat[i][j]
        m[i][3] = pos[i]
    return m


def mat44_by_pos_quat(pos, quat):
    """
    Construct a matrix using a position and orientation
    :param pos: list, position [x, y, z]
    :param quat: list, quaternion [x, y, z, w]
    :return: 4x4 matrix
    """
    quat_mat = mat33_by_quat(quat)
    return mat44_by_pos_mat33(pos, quat_mat)


def mat44_by_pos_abc(pos, abc):
    """
    Construct a matrix using position and orientation.
    :param pos: list, position [x, y, z]
    :param abc: list, Euler Angles [z, y', x"]
    :return: 4x4 matrix
    """
    abc_mat = mat33_by_abc(abc)
    return mat44_by_pos_mat33(pos, abc_mat)


def mat44_to_pos_abc(mat):
    """
    Construct position and orientation by matrix.
    :param mat: 4x4 matrix.
    :return: list, position [x, y, z], list, Euler Angles [z, y', x"]
    """
    return mat44_to_pos(mat), mat33_to_abc(mat)


def mat44_to_pos_quat(mat):
    """
    Construct a position and orientation using a matrix.
    :param mat: 4x4 matrix
    :return: list, position [x, y, z], list, quaternion [x, y, z, w]
    """
    pos = []
    quat_mat = numpy.eye(3, 3)
    for i in range(3):
        for j in range(3):
            quat_mat[i][j] = mat[i][j]
        pos.append(mat[i][3])
    quat = mat33_to_quat(quat_mat)
    return pos, quat


def mat44_to_pos(mat):
    """
    Get position from matrix.
    :param mat: 4x4 matrix
    :return: list, position
    """
    return [mat[i][3] for i in range(3)]


def get_relative_xform(mat_from, mat_to):
    """
    Get the relative transform between two matrices or, in other words, using
    the transformation from one matrix to another matrix.
    :param mat_from: 4x4 matrix.
    :param mat_to: 4x4 matrix
    :return: 4x4 matrix.
    """
    return numpy.matmul(mat_to, numpy.linalg.inv(mat_from))


def transform_mat(xform, mat):
    """
    Transform a matrix by another matrix.
    :param mat: 4x4 matrix
    :param xform: 4x4 matrix
    :return:
    """
    return numpy.matmul(xform, mat)


def transform_mat_from_to(mat, mat_from, mat_to):
    """
    Transform a matrix using the relative transform between two matrices, or,
    in other words, using the transformation from one matrix to another matrix.
    :param mat:
    :param mat_from:
    :param mat_to:
    :return:
    """
    xform = get_relative_xform(mat_from, mat_to)
    return transform_mat(xform, mat)


def get_f1_to_f2_xform(pose_from, pose_to):
    # util.prGreen('base pose: {}'.format(self.pose_from))
    # util.prGreen('member pose: {}'.format(self.pose_to))
    from_f1_to_f2 = get_relative_xform(mat44_by_pos_quat(pose_from[0], pose_from[1]),
                                        mat44_by_pos_quat(pose_to[0], pose_to[1]))
    return from_f1_to_f2
