import argparse
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
import cv2
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = FastSAM('/home/hengxuy/Downloads/FastSAM-x.pt')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
print(f'Model is running on: {DEVICE}')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    try:
        while True:
            start = time.perf_counter()
            suc, frame = cap.read()

            if not suc:
                print("Error: Could not read frame.")
                break
            everything_results = model(
                source=frame,
                device=DEVICE,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )
            prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
            # ann = prompt_process.everything_prompt()
            ann = prompt_process.text_prompt(text="the small grey cube object")
            img = prompt_process.plot_to_result(frame, annotations=ann)
            cv2.imshow('seg Stream', img)

            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()