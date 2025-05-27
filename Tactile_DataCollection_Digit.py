from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Joy  # Import the correct message type

from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
import pprint

ESC_KEY = 27

# Initialize ROS Node
rospy.init_node('digit_joystick_publisher', anonymous=True)
pub = rospy.Publisher('/digit_joystick', Joy, queue_size=10)

# DIGIT Sensors Settings
digits = DigitHandler.list_digits()
pprint.pprint(digits)

# Connect to DIGIT devices
digit0 = Digit("D21063", "Right Thumb")
digit1 = Digit("D21064", "Right Index")
digit2 = Digit("D21066", "Right Mid")
digit3 = Digit("D21068", "Right Ring")

for digit in [digit0, digit1, digit2, digit3]:
    digit.connect()
    digit.set_intensity(Digit.LIGHTING_MAX)
    digit.set_resolution(Digit.STREAMS["QVGA"])
    digit.set_fps(Digit.STREAMS["QVGA"]["fps"]["30fps"])

# Function to add a title to an image
def add_title(img, title, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
    cv2.putText(img, title, pos, font, font_scale, font_color, thickness, line_type)
    return img


def preprocess_frame_bilateral(frame):
    """
    Applies bilateral filtering to reduce high-frequency noise while preserving edges.

    Args:
        frame: Raw tactile image (RGB).

    Returns:
        preprocessed_frame: Noise-reduced grayscale image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter (d = 9, sigmaColor = 75, sigmaSpace = 75)
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    return gray_filtered


def normalize_pressure_absolute(pressure_value, baseline=100, alpha=0.0001):
    """
    Normalizes pressure using a sigmoid function, but uses absolute deviation from the baseline.

    Args:
        pressure_value: Raw pressure value.
        baseline: The baseline pressure value (default ~100).
        alpha: Sigmoid scaling factor (default 0.0001).

    Returns:
        Normalized pressure in range [0, 100].
    """
    deviation = abs(pressure_value - baseline)  # Ensure positive input to sigmoid
    sigmoid_value = (1 / (1 + np.exp(-alpha * deviation))) - (1 / (1 + np.exp(alpha * baseline)))

    # Scale the range from [0, max_val] to [0, 100]
    max_scaling_factor = (1 - (1 / (1 + np.exp(alpha * baseline))))
    normalized_pressure = (sigmoid_value / max_scaling_factor) * 100

    return normalized_pressure

def estimate_pressure_sobel_filtered(baseline_frame, curr_frame, threshold=100):
    """
    Estimates pressure using Sobel gradients but filters out small changes.
    Args:
        baseline_frame: Reference frame (no contact).
        curr_frame: Current tactile image.
        threshold: Minimum Sobel difference to consider contact.
    Returns:
        pressure_value: Estimated pressure after filtering small noise.
    """
    baseline_gray = preprocess_frame_bilateral(baseline_frame)
    curr_gray = preprocess_frame_bilateral(curr_frame)

    # Compute Sobel gradients
    sobel_x_base = cv2.Sobel(baseline_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y_base = cv2.Sobel(baseline_gray, cv2.CV_64F, 0, 1, ksize=5)

    sobel_x_curr = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y_curr = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=5)

    # Compute gradient magnitude difference
    grad_diff = np.sqrt((sobel_x_curr - sobel_x_base) ** 2 + (sobel_y_curr - sobel_y_base) ** 2)

    # Remove small noise below the threshold
    grad_diff[grad_diff < threshold] = 0

    # Compute sum of gradient magnitude after filtering
    pressure_value = np.sum(grad_diff)
    return pressure_value

if __name__ == '__main__':
    try:
        rate = rospy.Rate(30)  # 30 Hz loop
        count = 0
        ch_tri = 0
        ch_base = 0

        baseline_thumb = []
        baseline_index = []
        baseline_mid = []
        baseline_ring = []

        prev_frames = {
            "thumb": digit0.get_frame(),
            "index": digit1.get_frame(),
            "mid": digit2.get_frame(),
            "ring": digit3.get_frame(),
        }
        current_frames = None

        while not rospy.is_shutdown():
            count += 1
            if count >= 100 and ch_tri < 1:
                prev_frames = {
                    "thumb": digit0.get_frame(),
                    "index": digit1.get_frame(),
                    "mid": digit2.get_frame(),
                    "ring": digit3.get_frame(),
                }
                ch_tri = 1
                print("select Once")

            if count >= 1000:
                count = 0

            frame0 = digit0.get_frame()
            frame1 = digit1.get_frame()
            frame2 = digit2.get_frame()
            frame3 = digit3.get_frame()

            # Compute tactile changes
            pressure_thumb = estimate_pressure_sobel_filtered(prev_frames["thumb"], frame0)
            pressure_index = estimate_pressure_sobel_filtered(prev_frames["index"], frame1)
            pressure_mid = estimate_pressure_sobel_filtered(prev_frames["mid"], frame2)
            pressure_ring = estimate_pressure_sobel_filtered(prev_frames["ring"], frame3)

            if count >= 100 and count < 200 and ch_base < 1:
                baseline_thumb.append(pressure_thumb)
                baseline_index.append(pressure_index)
                baseline_mid.append(pressure_mid)
                baseline_ring.append(pressure_ring)
            if count >= 200 and ch_base < 1:
                ch_base = 1
                base_m_thumb = np.mean(baseline_thumb)
                base_m_index = np.mean(baseline_index)
                base_m_mid = np.mean(baseline_mid)
                base_m_ring = np.mean(baseline_ring)
                print("base mean once")


            if ch_base == 1:
                # Normalize pressure values (0-1 range)
                pressure_values = np.array([normalize_pressure_absolute(pressure_thumb, base_m_thumb),
                                            normalize_pressure_absolute(pressure_index, base_m_index),
                                            normalize_pressure_absolute(pressure_mid, base_m_mid),
                                            normalize_pressure_absolute(pressure_ring, base_m_ring)])

                # Generate joystick message
                joy_msg = Joy()
                joy_msg.header.stamp = rospy.Time.now()

                joy_msg.axes = [0.0] * 8  # Adjust size if needed
                joy_msg.buttons = [0] * 12  # Adjust size if needed
                joy_msg.axes[0] = pressure_values[0]  # Thumb pressure → Axis 0
                joy_msg.axes[1] = pressure_values[1]  # Index finger → Axis 1
                joy_msg.axes[2] = pressure_values[2]  # Middle finger → Axis 2
                joy_msg.axes[3] = pressure_values[3]  # Ring finger → Axis 3

                pub.publish(joy_msg)

            # Display tactile images
            frame0_titled = add_title(frame0, 'Right Thumb', (10, 30))
            frame1_titled = add_title(frame1, 'Right Index', (10, 30))
            frame2_titled = add_title(frame2, 'Right Mid', (10, 30))
            frame3_titled = add_title(frame3, 'Right Ring', (10, 30))

            top_row = cv2.hconcat([frame0_titled, frame1_titled])
            bottom_row = cv2.hconcat([frame2_titled, frame3_titled])
            combined_frame = cv2.vconcat([top_row, bottom_row])

            cv2.imshow('Tactile View', combined_frame)
            if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                break

            rate.sleep()  # Maintain loop rate

    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        cv2.destroyAllWindows()