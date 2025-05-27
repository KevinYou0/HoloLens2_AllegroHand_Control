## general import
from PIL import Image
import pdb
import os
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import multiprocessing as mp
import rospy

from pynput import keyboard

import multiprocessing as mp
import numpy as np
from numpy.linalg import norm, inv
import cv2
import viewer.hl2ss_imshow as hl2ss_imshow
import viewer.hl2ss as hl2ss
import viewer.hl2ss_lnm as hl2ss_lnm
import viewer.hl2ss_utilities as hl2ss_utilities
import viewer.hl2ss_mp as hl2ss_mp
import viewer.hl2ss_3dcv as hl2ss_3dcv
import viewer.hl2ss_sa as hl2ss_sa
from bitalino import BITalino

# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = "10.42.0.201"

# Camera parameters
# See etc/hl2_capture_formats.txt for a list of supported formats
pv_width     = 760
pv_height    = 428
pv_framerate = 30

# Marker properties
radius = 5
head_color  = (  0,   0, 255)
left_color  = (  0, 255,   0)
right_color = (255,   0,   0)
right_hand_color = (255,   255,   0)
gaze_color  = (255,   0, 255)
thickness = -1

# Buffer length in seconds
buffer_length = 5

# Spatial Mapping settings
triangles_per_cubic_meter = 1000
mesh_threads = 2
sphere_center = [0, 0, 0]
sphere_radius = 5

# emg settings
macAddress = "00:21:06:BE:26:EC"

# This example will collect data for 5 sec.
batteryThreshold = 30
acqChannels = [1, 2, 3]
samplingRate = 1000
nSamples = 10
digitalOutput_on = [1, 1]
digitalOutput_off = [0, 0]
# Connect to BITalino
device = BITalino(macAddress)
# Set battery threshold
device.battery(batteryThreshold)
device.start(samplingRate, acqChannels)

total_EMG = 30
#------------------------------------------------------------------------------
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
    angle = 3.14-angle_radians
    return angle

def get_joint_angles(point_positions):
    angle_list = []
    for i in range(1,len(point_positions)-1):
        # Calculate vectors BA and BC
        BA = point_positions[i-1] - point_positions[i]
        BC = point_positions[i+1] - point_positions[i]

        # Calculate the dot product of BA and BC
        dot_product = np.dot(BA, BC)

        # Calculate the magnitudes of BA and BC
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
        angle_list.append(3.14-angle_radians)
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
    angle_list.append(3.14-angle_radians)

    # joint 1
    angle_list.append(0.0)

    # joint 2
    BA = point_positions[1] - point_positions[3]
    BC = point_positions[5] - point_positions[3]
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
    angle_list.append(3.14-angle_radians)

    # joint 3
    BA = point_positions[3] - point_positions[4]
    BC = point_positions[5] - point_positions[4]
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    angle_radians = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
    angle_list.append(3.14-angle_radians)

    return angle_list

def calculate_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal/norm(normal)

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

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    # Set region of 3D space to sample
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)

    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()
    sm_manager.set_volumes(volumes)
    sm_manager.get_observed_surfaces()
    
    # Start PV and Spatial Input streams --------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, ...)
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, None)
    sink_pv.get_attach_response()
    sink_si.get_attach_response()

    #####################
    ###### ros set ######
    #####################
    rospy.init_node('hololens_angle_publisher', anonymous=True)
    angle_publisher = rospy.Publisher('hololens_joint_angles', Joy, queue_size=10)
    hand_pos_publisher = rospy.Publisher('/gripper_control', Joy, queue_size=10)

    # Main Loop ---------------------------------------------------------------
    try:
        count=0
        wait_time = 0
        shouwaice_total = 0
        shouneice_total = 0
        damuzhi_total = 0

        while enable and not rospy.is_shutdown():

            shouwaice = np.abs(np.squeeze(device.read(nSamples)[:, 5] - 500))
            shouneice = np.abs(np.squeeze(device.read(nSamples)[:, 6] - 500))
            damuzhi = np.abs(np.squeeze(device.read(nSamples)[:, 7] - 500))
            shouwaice_mean = np.mean(shouwaice)
            shouneice_mean = np.mean(shouneice)
            damuzhi_mean = np.mean(damuzhi)

            if (count<5):
                shouwaice_total += shouwaice_mean
                shouneice_total += shouneice_mean
                damuzhi_total += damuzhi_mean
                count += 1
            else:
                shouwaice_total_mean = shouwaice_total/count
                shouneice_total_mean = shouneice_total/count
                damuzhi_total_mean = damuzhi_total/count

                count = 0
                shouwaice_total = 0
                shouneice_total = 0
                damuzhi_total = 0

                total_EMG = shouwaice_total_mean + shouneice_total_mean + damuzhi_total_mean

            # env.render()
            # Download observed surfaces ------------------------------------------
            sm_manager.get_observed_surfaces()

            # Wait for PV frame ---------------------------------------------------
            sink_pv.acquire()

            # Get PV frame and nearest (in time) Spatial Input frame --------------
            _, data_pv = sink_pv.get_most_recent_frame()
            if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                continue

            _, data_si = sink_si.get_nearest(data_pv.timestamp)
            if (data_si is None):
                continue

            image = data_pv.payload.image
            si = hl2ss.unpack_si(data_si.payload)

            # Update PV intrinsics ------------------------------------------------
            # PV intrinsics may change between frames due to autofocus
            pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
            pv_extrinsics = np.eye(4, 4, dtype=np.float32)
            pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

            # Compute world to PV image transformation matrix ---------------------
            world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)

            angle_lists = []

            head_position = [0,0,0]
            head_forward = [1,0,0]
            head_up = [0,1,0]
            if (si.is_valid_head_pose()):
                head_pose = si.get_head_pose()
                head_position = [head_pose.position[0],
                                 head_pose.position[1],
                                 head_pose.position[2]]
                head_forward = head_pose.forward
                head_up = head_pose.up
            point_global = [0,0,0]

            # Assuming angle_lists is updated within the loop
            joy_msg = Joy()
            joy_msg.header.stamp = rospy.Time.now()
            joy_msg.axes = [0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0]  # Flatten the list if necessary
            joy_msg.buttons = []  # Optionally, you can also publish button states if relevant

            joy_msg_hand = Joy()
            joy_msg_hand.header.stamp = rospy.Time.now()
            joy_msg_hand.axes = [0, 0, 0, 0, 0, 0, 0]  # Flatten the list if necessary
            joy_msg_hand.buttons = []  # Optionally, you can also publish button states if relevant

            # Draw left Hand joints -----------------------------------------------
            if (si.is_valid_hand_right()):
                right_hand = si.get_hand_right()
                right_joints = hl2ss_utilities.si_unpack_hand(right_hand)
                point_global= [right_joints.positions[1][0],
                               right_joints.positions[1][1],
                               right_joints.positions[1][2]]
                thumb_positions = [right_joints.positions[0], right_joints.positions[1], right_joints.positions[2],
                                   right_joints.positions[3], right_joints.positions[4], right_joints.positions[5]]
                index_positions = [right_joints.positions[6], right_joints.positions[7], right_joints.positions[8], right_joints.positions[9], right_joints.positions[10]]
                middle_positions = [right_joints.positions[11], right_joints.positions[12], right_joints.positions[13], right_joints.positions[14], right_joints.positions[15]]
                ring_positions = [right_joints.positions[16], right_joints.positions[17], right_joints.positions[18], right_joints.positions[19], right_joints.positions[20]]

                thumb_angles = get_joint_angles_thumb(thumb_positions)
                index_angles = get_joint_angles(index_positions)
                middle_angles = get_joint_angles(middle_positions)
                ring_angles = get_joint_angles(ring_positions)

                point_local = transform_point_to_local(point_global, head_position, head_forward, head_up)
                angle_lists = [thumb_angles, index_angles, middle_angles, ring_angles, [-point_local[2], -point_local[0], -point_local[1] + 0.4]]
                right_image_points = hl2ss_3dcv.project(right_joints.positions, world_to_image)
                hl2ss_utilities.draw_points(image, right_image_points.astype(np.int32), radius, right_color, thickness)
                hl2ss_utilities.draw_points(image, right_image_points.astype(np.int32)[[0,1, 6, 21]], radius, right_hand_color, thickness)

                # Assuming angle_lists is updated within the loop
                joy_msg = Joy()
                joy_msg.header.stamp = rospy.Time.now()
                index_rot = get_finger_rotation_angles([right_joints.positions[17], right_joints.positions[7], right_joints.positions[10]])
                middle_rot = get_finger_rotation_angles([right_joints.positions[7], right_joints.positions[12], right_joints.positions[15]])
                ring_rot = get_finger_rotation_angles([right_joints.positions[7], right_joints.positions[17], right_joints.positions[20]])

                scale_para = 1.0 + scale_raw / 50.0
                #scale_para = 1
                print(scale_para)

                number_1 = 0.3;
                damuzhi_cuo_1 = 0.20;
                damuzhi_cuo_2 = 0.00;

                shizhi_nie_1 = number_1 + 0.15;
                shizhi_nie_2 = number_1;
                shizhi_nie_3 = number_1;

                zhongzhi_nie_1 = number_1 + 0.15;
                zhongzhi_nie_2 = number_1;
                zhongzhi_nie_3 = number_1;

                xiaozhi_nie_1 = number_1 + 0.15;
                xiaozhi_nie_2 = number_1;
                xiaozhi_nie_3 = number_1;

                zhua_2_mult = 1.0;
                zhua_3_mult = 2.0;
                zhua_4_mult = 3.0;

                if (total_EMG<40):
                    joy_msg.axes = [0.05016486161788896, -0.040153412958728756,
                                                           0.4905858588030592, -0.019793448452197562,
                                                           0.2916829866421884, -0.12656689880916988, 0.5976206351605217,
                                                           -0.11041210751704329,
                                                           0.02523891129763445, 0.009492744825731576,
                                                           0.6236970245584349, -0.025600961698601116,
                                                           1.4254833523289419, -0.06323080002774033, 0.0362781975699292,
                                                           0.0456977769148539]
                elif (total_EMG>=40 and total_EMG<55):
                    joy_msg.axes = [0.05016486161788896,
                                                           -0.040153412958728756 + shizhi_nie_1 * zhua_2_mult,
                                                           0.4905858588030592 + shizhi_nie_2 * zhua_2_mult,
                                                           -0.019793448452197562 + shizhi_nie_3 * zhua_2_mult,
                                                           0.2916829866421884,
                                                           -0.12656689880916988 + zhongzhi_nie_1 * zhua_2_mult,
                                                           0.5976206351605217 + zhongzhi_nie_2 * zhua_2_mult,
                                                           -0.11041210751704329 + zhongzhi_nie_3 * zhua_2_mult,
                                                           0.02523891129763445,
                                                           0.009492744825731576 + xiaozhi_nie_1 * zhua_2_mult,
                                                           0.6236970245584349 + xiaozhi_nie_2 * zhua_2_mult,
                                                           -0.025600961698601116 + xiaozhi_nie_3 * zhua_2_mult,
                                                           1.4254833523289419, -0.06323080002774033,
                                                           0.0362781975699292 + damuzhi_cuo_1 * zhua_2_mult,
                                                           0.0456977769148539 + damuzhi_cuo_2 * zhua_2_mult]
                elif (total_EMG >= 55 and total_EMG < 70):
                    joy_msg.axes = [0.05016486161788896,
                                                           -0.040153412958728756 + shizhi_nie_1 * zhua_3_mult,
                                                           0.4905858588030592 + shizhi_nie_2 * zhua_3_mult,
                                                           -0.019793448452197562 + shizhi_nie_3 * zhua_3_mult,
                                                           0.2916829866421884,
                                                           -0.12656689880916988 + zhongzhi_nie_1 * zhua_3_mult,
                                                           0.5976206351605217 + zhongzhi_nie_2 * zhua_3_mult,
                                                           -0.11041210751704329 + zhongzhi_nie_3 * zhua_3_mult,
                                                           0.02523891129763445,
                                                           0.009492744825731576 + xiaozhi_nie_1 * zhua_3_mult,
                                                           0.6236970245584349 + xiaozhi_nie_2 * zhua_3_mult,
                                                           -0.025600961698601116 + xiaozhi_nie_3 * zhua_3_mult,
                                                           1.4254833523289419, -0.06323080002774033,
                                                           0.0362781975699292 + damuzhi_cuo_1 * zhua_3_mult,
                                                           0.0456977769148539 + damuzhi_cuo_2 * zhua_3_mult]
                else:
                    joy_msg.axes = [0.05016486161788896,
                                                           -0.040153412958728756 + shizhi_nie_1 * zhua_4_mult,
                                                           0.4905858588030592 + shizhi_nie_2 * zhua_4_mult,
                                                           -0.019793448452197562 + shizhi_nie_3 * zhua_4_mult,
                                                           0.2916829866421884,
                                                           -0.12656689880916988 + zhongzhi_nie_1 * zhua_4_mult,
                                                           0.5976206351605217 + zhongzhi_nie_2 * zhua_4_mult,
                                                           -0.11041210751704329 + zhongzhi_nie_3 * zhua_4_mult,
                                                           0.02523891129763445,
                                                           0.009492744825731576 + xiaozhi_nie_1 * zhua_4_mult,
                                                           0.6236970245584349 + xiaozhi_nie_2 * zhua_4_mult,
                                                           -0.025600961698601116 + xiaozhi_nie_3 * zhua_4_mult,
                                                           1.4254833523289419, -0.06323080002774033,
                                                           0.0362781975699292 + damuzhi_cuo_1 * zhua_4_mult,
                                                           0.0456977769148539 + damuzhi_cuo_2 * zhua_4_mult]



                # joy_msg.axes = [(index_rot-1.5)*1.5 * scale_para, angle_lists[1][0] * scale_para, angle_lists[1][1] * scale_para, angle_lists[1][2] * scale_para,
                #                 (middle_rot-2)*1.5 * scale_para, angle_lists[2][0] * scale_para, angle_lists[2][1] * scale_para, angle_lists[2][2] * scale_para,
                #                 (ring_rot-1.8)*1.5 * scale_para, angle_lists[3][0] * scale_para, angle_lists[3][1] * scale_para, angle_lists[3][2] * scale_para,
                #                 (angle_lists[0][0]-1.2)*2 * scale_para, angle_lists[0][1] * scale_para, angle_lists[0][2] * scale_para, angle_lists[0][2] * scale_para]  # Flatten the list if necessary
                joy_msg.buttons = []  # Optionally, you can also publish button states if relevant


                # calculate the relative position to head
                right_hand_ref_point = np.squeeze(np.asarray(right_joints.positions[0]))
                head_right = np.cross(np.squeeze(np.asarray(head_up)), np.squeeze(np.asarray(head_forward)))
                rotation_matrix = np.column_stack((head_right, head_up, head_forward))
                hand_position_relative = right_hand_ref_point - np.squeeze(np.asarray(head_position))
                hand_in_head_coordinates = rotation_matrix.T.dot(hand_position_relative)

                # calculate the hand rotation
                points1 = np.squeeze(np.asarray(right_joints.positions[0]))
                points2 = np.squeeze(np.asarray(right_joints.positions[6]))
                points3 = np.squeeze(np.asarray(right_joints.positions[21]))

                return_up_norm = calculate_plane_normal(points1, points2, points3)
                return_forward_norm = ((np.squeeze(np.asarray(right_joints.positions[1])) - np.squeeze(np.asarray(right_joints.positions[0]))) /
                                       norm(np.squeeze(np.asarray(right_joints.positions[1])) - np.squeeze(np.asarray(right_joints.positions[0]))))
                return_left_norm = -np.cross(np.squeeze(np.asarray(return_up_norm)), np.squeeze(np.asarray(return_forward_norm)))

                ## rectified so that hand is 90 degree

                coord_A = np.array([
                    -head_right,
                    np.squeeze(np.asarray(head_up)),  # x-axis
                    np.squeeze(np.asarray(head_forward)) # z-axis
                ])
                # Define the axes of coordinate system B (columns are the x, y, z axes)
                coord_B = np.array([
                    return_left_norm,
                    return_up_norm, # x-axis
                    return_forward_norm,  # y-axis
                ])

                rotated_quaternion = compute_quaternion_from_coordinates(coord_A, coord_B)
                q_180_a = (np.cos(np.pi), np.sin(np.pi), 0, 0)
                q_a = (rotated_quaternion[0], rotated_quaternion[1], rotated_quaternion[2], rotated_quaternion[3])
                q_combined = quaternion_multiply(q_180_a, q_a)


                joy_msg_hand = Joy()
                joy_msg_hand.header.stamp = rospy.Time.now()
                joy_msg_hand.axes = [hand_in_head_coordinates[2], hand_in_head_coordinates[0], hand_in_head_coordinates[1]+0.5,
                                     q_combined[2], -q_combined[1], -q_combined[3], q_combined[0]]
                ### [3, 1, 2]
                ### [ , , front-back, w]
                joy_msg_hand.buttons = []  # Optionally, you can also publish button states if relevant

            angle_publisher.publish(joy_msg)
            hand_pos_publisher.publish(joy_msg_hand)
            rospy.Rate(100).sleep()
            # Display frame -------------------------------------------------------
            cv2.imshow('Video', image)
            cv2.waitKey(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        device.stop()
        device.close()
        print("Device connection properly closed.")
        # Stop Spatial Mapping data manager ---------------------------------------
        sm_manager.close()

        # Stop PV and Spatial Input streams ---------------------------------------
        sink_pv.detach()
        sink_si.detach()
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        ## Stop keyboard events ----------------------------------------------------
        listener.join()