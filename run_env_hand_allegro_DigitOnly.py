## general import
from PIL import Image
import pdb
import os
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import multiprocessing as mp

from pynput import keyboard

import multiprocessing as mp
import numpy as np
from numpy.linalg import norm, inv
import cv2
import rospy
from sensor_msgs.msg import Joy

from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
import pprint

# DIGITs Sensor Settings
digits = DigitHandler.list_digits()
pprint.pprint(digits)

# Connect to a Digit device with serial number with friendly name
digit0 = Digit("D21063", "Right Thumb")
digit0.connect()

digit1 = Digit("D21064", "Right Index")
digit1.connect()

digit2 = Digit("D21066", "Right Mid")
digit2.connect()

digit3 = Digit("D21068", "Right Ring")
digit3.connect()

# Change LED illumination intensity
digit0.set_intensity(Digit.LIGHTING_MAX)
digit1.set_intensity(Digit.LIGHTING_MAX)
digit2.set_intensity(Digit.LIGHTING_MAX)
digit3.set_intensity(Digit.LIGHTING_MAX)

# Change DIGIT resolution to QVGA
qvga_res = Digit.STREAMS["QVGA"]
digit0.set_resolution(qvga_res)
digit1.set_resolution(qvga_res)
digit2.set_resolution(qvga_res)
digit3.set_resolution(qvga_res)

# Change DIGIT FPS to 15fps
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit0.set_fps(fps_30)
digit1.set_fps(fps_30)
digit2.set_fps(fps_30)
digit3.set_fps(fps_30)

# Function to add a title to an image
def add_title(img, title, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
    cv2.putText(img, title, pos, font, font_scale, font_color, thickness, line_type)
    return img

# ------------------------------------------------------------------------------
def normalize(v):
    """ Normalize a vector """
    return v / norm(v)

def rotation_matrix_to_quaternion(R):
    """ Convert a rotation matrix to a quaternion """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def compute_quaternion_from_coordinates(coord_A, coord_B):
    """
    Compute the quaternion that rotates coordinate system A to coordinate system B.
    coord_A and coord_B are 3x3 matrices where columns are the x, y, z axes.
    """
    # Normalize the axes to ensure they are orthogonal unit vectors
    coord_A = np.array([normalize(axis) for axis in coord_A.T]).T
    coord_B = np.array([normalize(axis) for axis in coord_B.T]).T

    # Compute the rotation matrix from A to B
    R_A_inv = inv(coord_A)
    R = np.dot(coord_B, R_A_inv)

    # Convert the rotation matrix to a quaternion
    quaternion = rotation_matrix_to_quaternion(R)

    return quaternion


def compute_right_vector(upward, forward):
    return np.cross(upward, forward)


def construct_rotation_matrix(forward, up, right):
    return np.array([right, up, -forward]).T


def transform_point_to_local(point_global, headset_position, headset_forward, headset_up):
    # Compute the right vector to form a complete basis
    headset_up_vec = np.array([headset_up[0], headset_up[1], headset_up[2]])
    headset_forward_vec = np.array([headset_forward[0], headset_forward[1], headset_forward[2]])
    headset_position_vec = np.array([headset_position[0], headset_position[1], headset_position[2]])
    right = compute_right_vector(headset_up_vec, headset_forward_vec)
    # Construct the rotation matrix using the headset's orientation
    rotation_matrix = construct_rotation_matrix(headset_forward_vec, headset_up_vec, right)
    # Translate the global point to the headset's origin
    translated_point = point_global - headset_position_vec
    # Apply the rotation matrix to align the point with the headset's orientation
    point_local = np.dot(rotation_matrix, translated_point)
    return point_local


def get_finger_rotation_angles(point_positions):
    angle = 0
    BA = point_positions[0] - point_positions[1]
    BC = point_positions[2] - point_positions[1]

    # Calculate the dot product of BA and BC
    dot_product = np.dot(BA, BC)

    # Calculate the magnitudes of BA and BC
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
    angle = 3.14 - angle_radians
    return angle


def get_joint_angles(point_positions):
    angle_list = []
    for i in range(1, len(point_positions) - 1):
        # Calculate vectors BA and BC
        BA = point_positions[i - 1] - point_positions[i]
        BC = point_positions[i + 1] - point_positions[i]

        # Calculate the dot product of BA and BC
        dot_product = np.dot(BA, BC)

        # Calculate the magnitudes of BA and BC
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
        angle_list.append(3.14 - angle_radians)
    return angle_list


def get_joint_angles_thumb(point_positions):
    angle_list = []

    # joint 0
    BA = point_positions[0] - point_positions[2]
    BC = point_positions[3] - point_positions[2]
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
    angle_list.append(3.14 - angle_radians)

    # joint 1
    angle_list.append(0.0)

    # joint 2
    BA = point_positions[1] - point_positions[3]
    BC = point_positions[5] - point_positions[3]
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
    angle_list.append(3.14 - angle_radians)

    # joint 3
    BA = point_positions[3] - point_positions[4]
    BC = point_positions[5] - point_positions[4]
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
    angle_list.append(3.14 - angle_radians)

    return angle_list


def calculate_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal / norm(normal)


def rotate_vector(vector, axis, angle_degrees):
    # Normalize the rotation axis
    axis = axis / norm(axis)

    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Compute the quaternion components
    w = np.cos(angle_radians / 2)
    x, y, z = axis * np.sin(angle_radians / 2)
    rotation_quaternion = np.array([w, x, y, z])

    # Convert the vector to a quaternion (with zero scalar part)
    vector_quaternion = np.array([0, vector[0], vector[1], vector[2]])

    # Quaternion conjugate
    rotation_conjugate = np.array([w, -x, -y, -z])

    # Perform the quaternion multiplication: q * v * q_conjugate
    rotated_vector_quaternion = quaternion_multiply(
        quaternion_multiply(rotation_quaternion, vector_quaternion),
        rotation_conjugate
    )

    # Extract the rotated vector part (x, y, z)
    rotated_vector = rotated_vector_quaternion[1:]
    return rotated_vector


def quaternion_multiply(q1, q2):
    # Perform quaternion multiplication
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


if __name__ == '__main__':
    ####################
    ## hololens settings
    ####################
    enable = True
    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable


    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Main Loop ---------------------------------------------------------------
    try:
        count = 0
        while enable and not rospy.is_shutdown():

            ## show DIGITs sensor frames
            frame0 = digit0.get_frame()
            frame1 = digit1.get_frame()
            frame2 = digit2.get_frame()
            frame3 = digit3.get_frame()

            # Add titles to each frame
            frame0_titled = add_title(frame0, 'Right Thumb', (10, 30))
            frame1_titled = add_title(frame1, 'Right Index', (10, 30))
            frame2_titled = add_title(frame2, 'Right Mid', (10, 30))
            frame3_titled = add_title(frame3, 'Right Ring', (10, 30))

            # Concatenate images horizontally to form the top and bottom rows
            top_row = cv2.hconcat([frame0_titled, frame1_titled])
            bottom_row = cv2.hconcat([frame2_titled, frame3_titled])
            combined_frame = cv2.vconcat([top_row, bottom_row])
            cv2.imshow('Combined', combined_frame)

            cv2.waitKey(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:


        ## Stop keyboard events ----------------------------------------------------
        listener.join()