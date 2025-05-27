## general import
from PIL import Image
import pdb
import os
import pickle
import multiprocessing as mp
import time
import multiprocessing as mp

from pynput import keyboard

import multiprocessing as mp
import numpy as np
from numpy.linalg import norm, inv
from bitalino import BITalino

# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = "10.42.0.201"

# Camera parameters
# See etc/hl2_capture_formats.txt for a list of supported formats
pv_width = 760
pv_height = 428
pv_framerate = 30

# Marker properties
radius = 5
head_color = (0, 0, 255)
left_color = (0, 255, 0)
right_color = (255, 0, 0)
right_hand_color = (255, 255, 0)
gaze_color = (255, 0, 255)
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
batteryThreshold = 10
acqChannels = [1, 2 ,3, 4]
samplingRate = 1000
nSamples = 10
digitalOutput_on = [1, 1]
digitalOutput_off = [0, 0]
# Connect to BITalino
device = BITalino(macAddress)
# Set battery threshold
device.battery(batteryThreshold)
device.start(samplingRate, acqChannels)

if __name__ == '__main__':

    # Main Loop ---------------------------------------------------------------
    try:
        count = 0
        shouwaice_total = 0
        shouneice_total = 0
        damuzhi_total = 0
        while 1:
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

                print("wai: ")
                print(shouwaice_total_mean)
                print("nei: ")
                print(shouneice_total_mean)
                print("da: ")
                print(damuzhi_total_mean)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        device.stop()
        device.close()
        print("Device connection properly closed.")
