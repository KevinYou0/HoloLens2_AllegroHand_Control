import logging
import pprint
import time

import cv2

from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler

logging.basicConfig(level=logging.DEBUG)

# Print a list of connected DIGIT's
digits = DigitHandler.list_digits()
print("Connected DIGIT's to Host:")
pprint.pprint(digits)

# Connect to a Digit device with serial number with friendly name
digit0 = Digit("D21063", "Right Thumb")
digit0.connect()

digit1 = Digit("D21064", "Right Index")
digit1.connect()

digit2 = Digit("D21066", "Right Mid")
digit2.connect()

digit3 = Digit("D21068", "Right Ring")
digit3.connect()

# Print device info
print(digit1.info())

# Change LED illumination intensity
digit0.set_intensity(Digit.LIGHTING_MAX)
digit1.set_intensity(Digit.LIGHTING_MAX)
digit2.set_intensity(Digit.LIGHTING_MAX)
digit3.set_intensity(Digit.LIGHTING_MAX)

# Change DIGIT resolution to QVGA
qvga_res = Digit.STREAMS["QVGA"]
digit0.set_resolution(qvga_res)
digit1.set_resolution(qvga_res)
digit2.set_resolution(qvga_res)
digit3.set_resolution(qvga_res)

# Change DIGIT FPS to 15fps
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit0.set_fps(fps_30)
digit1.set_fps(fps_30)
digit2.set_fps(fps_30)
digit3.set_fps(fps_30)

# Function to add a title to an image
def add_title(img, title, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
    cv2.putText(img, title, pos, font, font_scale, font_color, thickness, line_type)
    return img

frame0_ori = digit0.get_frame()
frame1_ori = digit1.get_frame()
frame2_ori = digit2.get_frame()
frame3_ori = digit3.get_frame()

while 1:
    frame0 = digit0.get_frame()
    frame1 = digit1.get_frame()
    frame2 = digit2.get_frame()
    frame3 = digit3.get_frame()

    # frame0 = frame0 - frame0_ori
    # frame1 = frame1 - frame1_ori
    # frame2 = frame2 - frame2_ori
    # frame3 = frame3 - frame3_ori

    # Add titles to each frame
    frame0_titled = add_title(frame0, 'Right Thumb', (10, 30))
    frame1_titled = add_title(frame1, 'Right Index', (10, 30))
    frame2_titled = add_title(frame2, 'Right Mid', (10, 30))
    frame3_titled = add_title(frame3, 'Right Ring', (10, 30))

    # Concatenate images horizontally to form the top and bottom rows
    top_row = cv2.hconcat([frame0_titled, frame1_titled])
    bottom_row = cv2.hconcat([frame2_titled, frame3_titled])
    combined_frame = cv2.vconcat([top_row, bottom_row])

    cv2.imshow('Combined', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# # Display stream obtained from DIGIT
# digit1.show_view()
#
# # Disconnect DIGIT stream

digit0.disconnect()
digit1.disconnect()
digit2.disconnect()
digit3.disconnect()
cv2.destroyAllWindows()

# # Find a Digit by serial number and connect manually
# cap1 = cv2.VideoCapture(digit1["dev_name"])
# cap2 = cv2.VideoCapture(digit1["dev_name"])
# cap3 = cv2.VideoCapture(digit1["dev_name"])
#
#
# if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
#     print("Error: Could not open video capture.")
# else:
#     try:
#         while True:
#             suc1, frame1 = cap1.read()
#             suc2, frame2 = cap2.read()
#             suc3, frame3 = cap3.read()
#
#             if not suc1 or not suc2 or not suc3:
#                 print("Error: Could not read frame.")
#                 break
#             cv2.imshow('Right Index', frame1)
#             cv2.imshow('Right Mid', frame2)
#             cv2.imshow('Right Ring', frame3)
#
#             # Check for 'q' key to exit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     finally:
#         # Cleanup
#         cap1.release()
#         cap2.release()
#         cap3.release()
#         cv2.destroyAllWindows()