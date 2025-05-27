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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FastSAM('/home/hengxuy/Downloads/FastSAM-x.pt')

current_image = None
current_seg = None
new_processed_available = False
lock_image = threading.Lock()
lock_processed = threading.Lock()
keep_running = True
previous_input = None

#def capture_user_input():
#    global current_input
#    while True:
#        with lock_input:
#            current_input = input("Please type a sentence: ")
def plot_temp(image, annotations):
    if isinstance(annotations[0], dict):
        annotations = [annotation['segmentation'] for annotation in annotations]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h = image.shape[0]
    original_w = image.shape[1]

    fig = plt.gcf()

    if DEVICE == 'cpu':
        annotations = np.array(annotations)
    else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()
    print(annotations.shape)
    print(annotations)
    img_array = np.frombuffer(image, dtype=np.uint8).reshape(original_h, original_w, 3)
    result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return result


def process_images():
    global current_image, current_seg, new_processed_available, previous_input
    while keep_running:
        with lock_image:
            local_image = current_image

        if local_image is not None and previous_input is not None:
            frame = local_image
            everything_results = model(
                source=frame,
                device=DEVICE,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )

            prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
            ann = prompt_process.text_prompt(text=previous_input)
            current_seg = ann
            new_processed_available = True
            img = prompt_process.plot_to_result(frame, annotations=ann)
            cv2.imshow('seg Stream', img)


def main_loop():
    global current_image, current_seg, keep_running, new_processed_available, previous_input
    cap = cv2.VideoCapture(4)  # Assuming you're capturing from the default camera

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                with lock_image:
                    current_image = frame
                img = frame
                cv2.imshow('Camera Stream', img)

            current_input = input("Please type a sentence: ")
            if current_input != previous_input:
                previous_input = current_input
                with lock_processed:
                    new_processed_available = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        keep_running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start image processing thread
    processing_thread = threading.Thread(target=process_images, daemon=True)
    processing_thread.start()

    # Execute the main camera capturing loop
    main_loop()

    # Join threads before exiting to ensure they finish
    processing_thread.join()