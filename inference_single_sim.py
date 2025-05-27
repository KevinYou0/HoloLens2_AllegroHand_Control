import gymnasium
import gym
import numpy as np
import time
import gym_panda
import pybullet_data
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import numpy as np
import cv2
import time
import os
import socket
import pdb
import threading
import queue

target_queue = queue.Queue()
exit_program = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FastSAM('D:/Python_Project/FastSAM/models/FastSAM-x.pt')

current_input = None
current_image = None
current_depth = None
new_processed_available = False
keep_running = True

def transfer_from_mask(mask, depth_image, viewMatrix):
    ## calculate the camera depth
    height = 480
    weight = 640
    fov = 60.0
    def compute_focal_lengths(width, height, fov):
        fx = width / (2 * np.tan(np.radians(fov) / 2))
        fy = height / (2 * np.tan(np.radians(fov) / 2))
        return fx, fy

    fx, fy = compute_focal_lengths(weight, height, fov)
    object_depths = depth_image[mask == True]
    y_indices, x_indices = np.where(mask == 1)
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)

    representative_depth = np.median(object_depths)
    X_cam = (center_x - weight / 2) * representative_depth / fx
    Y_cam = (center_y - height / 2) * representative_depth / fy
    Z_cam = representative_depth

    ## Transform to World Frame
    inv_viewMatrix = np.linalg.inv(np.array(viewMatrix).reshape(4, 4))
    point_camera_frame = np.array([[X_cam], [Y_cam], [Z_cam], [1]])
    point_world_frame = np.dot(inv_viewMatrix, point_camera_frame)
    X_world, Y_world, Z_world, _ = point_world_frame.flatten()
    world_location = [X_world, Y_world, Z_world]
    return world_location

def main_loop():
    env = gym.make('panda-v0')
    observation, info = env.reset()
    env.render()

    done = False
    error = 0.01
    fingers = 1

    k_p = 10
    k_d = 1
    dt = 1. / 240.  # the default timestep in pybullet is 240 Hz
    while True:
        # for every step initializa the environment
        observation, info = env.reset()
        env.render()
        current_input = input("Please type a sentence: ")
        print("get input")
        frame = info['scene_rgb']
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
        mask = np.asarray(ann[0])

        obj_world_coord = transfer_from_mask(mask, info['scene_depth'], info['viewMatrix'])
        print("the predit target is:" + str(obj_world_coord))
        img = prompt_process.plot_to_result(frame, annotations=ann)
        cv2.imshow('img', img)


        print("finish rendering")
        # target_queue.put(obj_world_coord)  # Put the target's location in the queue
        # info['object_position'] = target_queue.get()

        for t in range(200):
            env.render()
            if "green" in current_input:
                info['object_position'] = env.list_object_pos[0]
            if "yellow" in current_input:
                info['object_position'] = env.list_object_pos[1]
            if "red" in current_input:
                info['object_position'] = env.list_object_pos[2]

            dx = info['object_position'][0] - observation[0]
            dy = info['object_position'][1] - observation[1]
            target_z = info['object_position'][2]
            if abs(dx) < error and abs(dy) < error and abs(dz) < error:
                fingers = 0
            if (observation[3] + observation[4]) < error + 0.02 and fingers == 0:
                target_z = 0.5
            dz = target_z - observation[2]
            pd_x = k_p * dx + k_d * dx / dt
            pd_y = k_p * dy + k_d * dy / dt
            pd_z = k_p * dz + k_d * dz / dt
            action = [pd_x, pd_y, pd_z, fingers]
            observation, reward, done, info = env.step(action)
            if done:
                print("finished")
                break

if __name__ == "__main__":
    main_loop()