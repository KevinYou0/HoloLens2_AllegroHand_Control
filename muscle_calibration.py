import numpy as np

DAMUZHI_MIN = 500.0
DAMUZHI_MAX = 546.8

SHOUWAICE_MIN = 500.0
SHOUWAICE_MAX = 580.7

SHOUNEICE_MIN = 500.0
SHOUNEICE_MAX = 569.7

# ## general import
# from PIL import Image
# import pdb
# import os
# import pickle
# import multiprocessing as mp
# import time
# import multiprocessing as mp
#
# from pynput import keyboard
#
# import multiprocessing as mp
# import numpy as np
# from numpy.linalg import norm, inv
# from bitalino import BITalino
#
# # Settings --------------------------------------------------------------------
#
# # HoloLens 2 address
# host = "10.42.0.201"
#
# # Camera parameters
# # See etc/hl2_capture_formats.txt for a list of supported formats
# pv_width = 760
# pv_height = 428
# pv_framerate = 30
#
# # Marker properties
# radius = 5
# head_color = (0, 0, 255)
# left_color = (0, 255, 0)
# right_color = (255, 0, 0)
# right_hand_color = (255, 255, 0)
# gaze_color = (255, 0, 255)
# thickness = -1
#
# # Buffer length in seconds
# buffer_length = 5
#
# # Spatial Mapping settings
# triangles_per_cubic_meter = 1000
# mesh_threads = 2
# sphere_center = [0, 0, 0]
# sphere_radius = 5
#
# # emg settings
# macAddress = "00:21:06:BE:26:EC"
#
# # This example will collect data for 5 sec.
# batteryThreshold = 30
# acqChannels = [1, 2 ,3, 4]
# samplingRate = 1000
# nSamples = 10
# digitalOutput_on = [1, 1]
# digitalOutput_off = [0, 0]
# # Connect to BITalino
# device = BITalino(macAddress)
# # Set battery threshold
# device.battery(batteryThreshold)
# device.start(samplingRate, acqChannels)
#
# shouwaice_MAX = 0
# shouwaice_MIN = 1000000000
#
# shouneice_MAX = 0
# shouneice_MIN = 1000000000
#
# damuzhi_MAX = 0
# damuzhi_MIN = 100000000
#
# if __name__ == '__main__':
#
#     # Main Loop ---------------------------------------------------------------
#     try:
#         count = 0
#         shouwaice_total = 0
#         shouneice_total = 0
#         damuzhi_total = 0
#         while 1:
#             shouwaice = np.abs(np.squeeze(device.read(nSamples)[:, 5]))
#             shouneice = np.abs(np.squeeze(device.read(nSamples)[:, 6]))
#             damuzhi = np.abs(np.squeeze(device.read(nSamples)[:, 7]))
#
#             shouwaice_mean = np.mean(shouwaice)
#             shouneice_mean = np.mean(shouneice)
#             damuzhi_mean = np.mean(damuzhi)
#
#             if shouwaice_mean > shouwaice_MAX:
#                 shouwaice_MAX =  shouwaice_mean
#             if shouwaice_mean < shouwaice_MIN:
#                 shouwaice_MIN = shouwaice_mean
#
#             if shouneice_mean > shouneice_MAX:
#                 shouneice_MAX = shouneice_mean
#             if shouneice_mean < shouneice_MIN:
#                 shouneice_MIN = shouneice_mean
#
#             if damuzhi_mean > damuzhi_MAX:
#                 damuzhi_MAX = damuzhi_mean
#             if damuzhi_mean < damuzhi_MIN:
#                 damuzhi_MIN = damuzhi_mean
#
#             print(f"damuzhi Max:{damuzhi_MAX}, damuzhi MIN:{damuzhi_MIN}, "
#                   f"shouwaice MAX:{shouwaice_MAX}, shouwaice MIN:{shouwaice_MIN}, "
#                   f"shouneice MAX:{shouneice_MAX}, shouneice MIN{shouneice_MIN}")
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         device.stop()
#         device.close()
#         print("Device connection properly closed.")

