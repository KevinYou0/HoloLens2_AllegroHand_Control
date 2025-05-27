## general import
from PIL import Image
import pdb
import os
import pickle
import multiprocessing as mp
import time
import multiprocessing as mp

from pynput import keyboard
import rospy
from sensor_msgs.msg import Joy

import multiprocessing as mp
import numpy as np
from numpy.linalg import norm, inv
from bitalino import BITalino

DAMUZHI_MIN = 500.0
DAMUZHI_MAX = 546.8

SHOUWAICE_MIN = 500.0
SHOUWAICE_MAX = 580.7

SHOUNEICE_MIN = 500.0
SHOUNEICE_MAX = 569.7

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
key_pressed = None

def on_press(key):
    global key_pressed
    key_pressed = key.char

listener = keyboard.Listener(on_press=on_press)
listener.start()

def interval_index(x, maximum):
    """
    Given a value x and its maximum value,
    divide [0, maximum] into 5 equal intervals and
    return the 1-based interval index in which x falls.
    """
    # Handle edge cases (e.g., negative or greater than maximum, if needed)
    if x <= 0:
        return 1
    if x >= maximum:
        return 4

    # Calculate which of the 5 intervals x falls into
    idx = int(4 * x / maximum)  # This yields a value from 0 to 4 typically
    # idx is 5 if x == maximum, but we handled x == maximum above
    return idx

def combined_interval(ia, ib, ic):
    """
    1) Compute the interval indices (1..5) for a, b, c using A, B, C.
    2) If two (or three) interval indices match, return that value.
    3) Otherwise, if all three are different, return the rounded mean.
    """

    # Check if all three are the same
    if ia == ib == ic:
        return ia

    # Check if at least two match
    if ia == ib:
        return ia
    if ib == ic:
        return ib
    if ia == ic:
        return ia

    # If none match (all three different), return the integer-rounded mean
    mean_val = round((ia + ib + ic) / 3)
    return mean_val


if __name__ == '__main__':

    #####################
    ###### ros set ######
    #####################
    rospy.init_node('hololens_angle_publisher', anonymous=True)
    angle_publisher = rospy.Publisher('hololens_joint_angles', Joy, queue_size=10)
    emg_raw_publisher = rospy.Publisher('emg_raw', Joy, queue_size=10)
    emg_label_publisher = rospy.Publisher('emg_label', Joy, queue_size=10)

    # Main Loop ---------------------------------------------------------------
    try:
        count=0
        wait_time=0
        shouwaice_total = 0
        shouneice_total = 0
        damuzhi_total = 0

        while 1 and not rospy.is_shutdown():
            count += 1

            # Assuming angle_lists is updated within the loop
            joy_msg = Joy()
            joy_msg.header.stamp = rospy.Time.now()

            number_1 = 0.3;
            damuzhi_cuo_1 = 0.20;
            damuzhi_cuo_2 = 0.30;

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
            #
            # # print(f"total EMG:{total_EMG}, raw shouwaice:{shouwaice_mean}, "
            # #       f"raw shouneice:{shouneice_mean}, raw damuzhi:{damuzhi_mean}")
            # print(f"idx shouneice:{shouneice_idx}, idx shouwaice:{shouwaice_idx}, idx damuzhi:{damuzhi_idx}, idx total:{final_idx}")
            print(total_EMG)
            if (total_EMG<40):

            #if (key_pressed == 'a'):
                joy_msg.axes = [0.05016486161788896, -0.040153412958728756,
                                                       0.4905858588030592, -0.019793448452197562,
                                                       0.2916829866421884, -0.12656689880916988, 0.5976206351605217,
                                                       -0.11041210751704329,
                                                       0.02523891129763445, 0.009492744825731576,
                                                       0.6236970245584349, -0.025600961698601116,
                                                       1.4254833523289419, -0.06323080002774033, 0.0362781975699292,
                                                       0.0456977769148539]
            elif (total_EMG>=40 and total_EMG<55):
            #elif (key_pressed == 's'):
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
            #elif (key_pressed == 'd'):
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
            elif (total_EMG >= 70):
            #elif (key_pressed == 'f'):
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
            else:
                joy_msg.axes = [0.05016486161788896, -0.040153412958728756,
                                                       0.4905858588030592, -0.019793448452197562,
                                                       0.2916829866421884, -0.12656689880916988, 0.5976206351605217,
                                                       -0.11041210751704329,
                                                       0.02523891129763445, 0.009492744825731576,
                                                       0.6236970245584349, -0.025600961698601116,
                                                       1.4254833523289419, -0.06323080002774033, 0.0362781975699292,
                                                       0.0456977769148539]

            ## rectified so that hand is 90 degree

            angle_publisher.publish(joy_msg)
            rospy.Rate(100).sleep()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # device.stop()
        # device.close()
        print("Device connection properly closed.")