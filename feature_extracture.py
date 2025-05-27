import gymnasium
import gym_panda
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pybullet_data
from PIL import Image
import cv2
import pdb
import os
import pickle
import glob
import transformers
from natsort import natsorted

directory_img = "./ViT/images"
directory_force = "./ViT/force"

if not os.path.exists(directory_img):
    os.makedirs(directory_img)
if not os.path.exists(directory_force):
    os.makedirs(directory_force)

file_list = glob.glob(os.path.join(directory_img, '*.ply'))
file_list = natsorted(file_list)

img_list = []
force_list = []

for file in file_list:
    image = cv2.imread(file)
    img_list.append(image)

file_list = glob.glob(os.path.join(directory_force, '*.pkl'))
file_list = natsorted(file_list)
for file in file_list:
    with open(file, 'rb') as file:
        my_object = pickle.load(file)
        force_list.append(my_object)

        # Normalize depth image to 0-255 and convert to 8-bit (if necessary)
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_8bit = np.uint8(depth_image_normalized)

        img_name = 'rgb_img_' + str(i_episode*200 + t)
        depth_name = 'depth_img_' + str(i_episode*200 + t)
        joint_name = 'joint_' + str(i_episode*200 + t)

        cv2.imwrite(os.path.join(directory_img, img_name + '.jpg'), rgb_image)
        cv2.imwrite(os.path.join(directory_img, depth_name + '.jpg'), depth_image_8bit)
        # Load the list of dictionaries back
        with open(os.path.join(directory_force, joint_name), 'wb') as f:
            pickle.dump(info['force_info'], f)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break