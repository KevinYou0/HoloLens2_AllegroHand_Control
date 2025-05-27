import argparse
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
import cv2
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cap = cv2.VideoCapture(0)
desired_width = 1920
desired_height = 2560
# x, y, w, h = 590, 430, 100, 100  # Example values
# # Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    try:
        while True:
            suc, frame = cap.read()

            if not suc:
                print("Error: Could not read frame.")
                break
            cv2.imshow('Largest Bright Area', frame)

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # largest_contour = max(contours, key=cv2.contourArea) if contours else None
            #
            # if largest_contour is not None:
            #     x, y, w, h = cv2.boundingRect(largest_contour)
            #     # Optionally draw the rectangle on the image to visualize
            #     largest_bright_area = frame[y:y + h, x:x + w]
            #     original_size = (frame.shape[1], frame.shape[0])  # (width, height)
            #     resized_image = cv2.resize(largest_bright_area, original_size, interpolation=cv2.INTER_CUBIC)
            #     cv2.imshow('Largest Bright Area', resized_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()