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
import threading
import pdb
import rospy
from sensor_msgs.msg import Joy, Image as ROSImage, PointCloud2, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_input = None
new_input_available = False
segmented_ann = None
segmented_prompt = None
def get_user_input():
    global current_input, new_input_available
    while True:
        input_text = input("Please type a sentence: ")
        current_input = input_text
        new_input_available = True

model = FastSAM('/home/hengxuy/Downloads/FastSAM-x.pt')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
print(f'Model is running on: {DEVICE}')

rgbd_image = None
depth_image = None
camera_info = None
bridge = CvBridge()

def rgbd_depth_image_callback(msg):
    global rgbd_image
    try:
        rgbd_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print(e)

def depth_image_callback(msg):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print(e)

def info_callback(data):
    global camera_info
    camera_info = data

def main():
    global current_input, new_input_available, segmented_ann, segmented_prompt
    rospy.init_node('mean_xyz_publisher', anonymous=True)
    joy_pub = rospy.Publisher('/user_detect', Joy, queue_size=10)
    point_cloud_pub = rospy.Publisher("/cal_points", PointCloud2, queue_size=10)
    rospy.Subscriber('/k4a/rgb_to_depth/image_raw', ROSImage, rgbd_depth_image_callback)
    rospy.Subscriber('/k4a/depth/image_raw', ROSImage, depth_image_callback)
    rospy.Subscriber('/k4a/depth/camera_info', CameraInfo, info_callback)
    rate = rospy.Rate(100)

    target_point = None
    none_pub = True
    while not rospy.is_shutdown():
        if rgbd_image is None or depth_image is None or camera_info is None:
            continue
        frame = rgbd_image[:,:,:3]
        depth_frame = depth_image
        height, width = depth_frame.shape
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        point_cloud_array = np.zeros((height, width, 3), dtype=np.float32)
        for v in range(height):
            for u in range(width):
                z = depth_frame[v, u] / 1000.0  # Assuming depth is in millimeters
                if z == 0:
                    continue  # Skip no depth info
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                point_cloud_array[v, u] = [x, y, z]

        if new_input_available:
            none_pub = True
            everything_results = model(
                source=frame,
                device=DEVICE,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )
            prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
            ann = prompt_process.text_prompt(text=current_input)

            segmented_ann = ann
            segmented_prompt = prompt_process
            new_input_available = False

        if segmented_ann is not None and none_pub:
            transparency_level = 0.6
            color = (30, 144, 255)
            mask = segmented_ann.squeeze()
            mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
            filtered_points = point_cloud_array * mask_3d
            filtered_points = filtered_points.reshape(-1, 3)
            non_zero_points = filtered_points[~np.all(filtered_points == 0, axis=1)]
            mean_xyz = np.mean(non_zero_points, axis=0)
            print(mean_xyz)
            joy_msg = Joy()
            joy_msg.header = Header(stamp=rospy.Time.now())
            joy_msg.axes = mean_xyz.tolist()
            joy_pub.publish(joy_msg)
            none_pub = False

            # Create the color overlay
            color_overlay = np.zeros_like(frame, dtype=np.float32)
            color_overlay[..., 0] = color[0]
            color_overlay[..., 1] = color[1]
            color_overlay[..., 2] = color[2]

            # Apply the mask to the color overlay
            color_overlay = color_overlay * mask_3d

            # Blend the original image and the color overlay using the mask
            alpha_channel = mask.astype(np.float32) * transparency_level
            blended = frame.astype(np.float32) * (1 - alpha_channel[..., None]) + color_overlay * alpha_channel[..., None]
            blended = blended.astype(np.uint8)

            # Add alpha channel to the blended image
            alpha = (1 - alpha_channel) * 255
            result = np.dstack((blended, alpha.astype(np.uint8)))

            img = result

        else:
            img = frame
        rate.sleep()
        cv2.imshow('seg Stream', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_thread = threading.Thread(target=get_user_input, daemon=True)
    input_thread.start()
    main()