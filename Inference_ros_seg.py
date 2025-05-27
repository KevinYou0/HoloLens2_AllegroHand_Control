import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image

from utils.tools import convert_box_xywh_to_xyxy
import numpy as np
import cv2
import time
import os
import msgpack
import socket
import pdb
import threading
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def callback(data):
    bridge = CvBridge()
    try:
        # Convert the ROS image into an OpenCV-compatible format
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    # Display the converted image
    cv2.imshow("Kinect Image", cv_image)
    cv2.waitKey(3)  # A small delay so that OpenCV can handle the display window events

def listener():
    # Initialize the ROS node
    rospy.init_node('kinect_image_listener', anonymous=True)

    # Subscribe to the Kinect image topic
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)

    # Keep the program alive until it is shut down
    rospy.spin()

if __name__ == '__main__':
    listener()