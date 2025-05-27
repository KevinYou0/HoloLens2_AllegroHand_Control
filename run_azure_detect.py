import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import pdb
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

import argparse
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from utils.tools import convert_box_xywh_to_xyxy
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = FastSAM('/home/hengxuy/Downloads/FastSAM-x.pt')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def callback(data):
    bridge = CvBridge()

    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    everything_results = model(
        source=cv_image,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(cv_image, everything_results, device=DEVICE)
    for box in everything_results[0].boxes:
        box = box.xyxy.cpu().numpy()[0]
        print(box)
        cv2.rectangle(cv_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    # ann = prompt_process.everything_prompt()
    # img = prompt_process.plot_to_result(cv_image, annotations=ann)
    cv2.imshow("Kinect Image", cv_image)
    cv2.waitKey(1)

def listener():
    # Initialize the ROS node
    rospy.init_node('kinect_image_listener', anonymous=True)

    # Subscribe to the Kinect image topic
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)

    # Keep the program alive until it is shut down
    rospy.spin()

if __name__ == '__main__':
    listener()