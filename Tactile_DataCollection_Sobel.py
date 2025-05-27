from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Joy  # Import the correct message type
from scipy.interpolate import splprep, splev

from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
import pprint
from datetime import datetime

ESC_KEY = 27
minimun_hull0 = 1000
minimun_hull1 = 1000
minimun_hull2 = 1000
minimun_hull3 = 1000


emg_raw_data = np.ones(3)
emg_label_data = np.ones(4)

# DIGIT Sensors Settings
digits = DigitHandler.list_digits()
pprint.pprint(digits)

# Connect to DIGIT devices
digit0 = Digit("D21063", "Right Thumb")
digit1 = Digit("D21064", "Right Index")
digit2 = Digit("D21066", "Right Mid")
digit3 = Digit("D21068", "Right Ring")


def emg_raw_callback(msg):
    timestamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
    emg_raw_data[0] = msg.axes[0]
    emg_raw_data[1] = msg.axes[1]
    emg_raw_data[2] = msg.axes[2]

def emg_label_callback(msg):
    timestamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()

    emg_label_data[0] = msg.axes[0]
    emg_label_data[1] = msg.axes[1]
    emg_label_data[2] = msg.axes[2]
    emg_label_data[3] = msg.axes[3]

for digit in [digit0, digit1, digit2, digit3]:
    digit.connect()
    digit.set_intensity(Digit.LIGHTING_MAX)
    digit.set_resolution(Digit.STREAMS["QVGA"])
    digit.set_fps(Digit.STREAMS["QVGA"]["fps"]["30fps"])

# Function to add a title to an image
def add_title(img, title, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
    cv2.putText(img, title, pos, font, font_scale, font_color, thickness, line_type)
    return img


def create_data_directories(base_dir="/media/hengxuy/Eric Du SSD"):
    """
    Creates a new trial folder with four subfolders:
    images1, images2, vectors1, vectors2.

    Returns a dictionary of the subfolder paths for easy access.
    """
    if base_dir is None:
        # Default to the current working directory if none is given
        base_dir = os.getcwd()

    # Use current date/time as part of folder name for uniqueness
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(base_dir, f"trial_{time_str}")

    # Create the top-level trial directory
    os.makedirs(trial_dir, exist_ok=True)

    # Define your four subfolders
    subfolders = ["rgb_tactile", "grey_tactile", "EDA", "CCI", "convex_area", "emg_raw", "emg_label"]
    subfolder_paths = {}

    for sub in subfolders:
        path = os.path.join(trial_dir, sub)
        os.makedirs(path, exist_ok=True)
        subfolder_paths[sub] = path

    return subfolder_paths


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

def normalize_pressure_absolute_CCI(pressure_value, baseline=100, upper_bound=600, midpoint=300, alpha=0.01):

    pressure_value = np.array(pressure_value, dtype=float)

    # Define the logistic function with a midpoint of 300.
    L = lambda x: 1 / (1 + np.exp(-alpha * (x - midpoint)))

    # Evaluate the logistic function at the baseline and upper_bound.
    L_baseline = L(baseline)
    L_upper = L(upper_bound)

    # Compute the logistic value for the given pressure_value.
    logistic_value = L(pressure_value)

    # Scale so that at x=baseline we get 0 and at x=upper_bound we get 100.
    normalized_pressure = (logistic_value - L_baseline) / (L_upper - L_baseline)

    return logistic_value

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
    normalized_pressure = sigmoid_value / max_scaling_factor

    return normalized_pressure

def estimate_pressure_sobel_filtered(baseline_frame, curr_frame, threshold=80):
    """
    Estimates pressure using Sobel gradients but filters out small changes.
    Args:
        baseline_frame: Reference frame (no contact).
        curr_frame: Current tactile image.
        threshold: Minimum Sobel difference to consider contact.
    Returns:
        pressure_value: Estimated pressure after filtering small noise.
    """

    global CCI_max
    baseline_gray = preprocess_frame_bilateral(baseline_frame)
    curr_gray = preprocess_frame_bilateral(curr_frame)

    # Compute Sobel gradients
    sobel_x_base = cv2.Sobel(baseline_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y_base = cv2.Sobel(baseline_gray, cv2.CV_64F, 0, 1, ksize=5)

    sobel_x_curr = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y_curr = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=5)

    # Compute gradient magnitude difference
    grad_diff = np.sqrt((sobel_x_curr - sobel_x_base) ** 2 + (sobel_y_curr - sobel_y_base) ** 2)
    P_max = np.max(grad_diff)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grad_diff)

    maxloc_new = (240 - maxLoc[0], maxLoc[1])
    # Remove small noise below the threshold
    grad_diff[grad_diff < threshold] = 0
    mask = grad_diff
    EDA = np.sum(mask)
    CCI = P_max / (EDA if EDA > 0 else 1)

    CCI_norm = normalize_pressure_absolute_CCI(P_max)
    # Compute sum of gradient magnitude after filtering
    pressure_value = np.sum(grad_diff)
    return pressure_value, mask, CCI_norm, P_max, maxloc_new

def estimate_fine_largest_contour(img):
    """
    Estimate the most stable and continuous contour from an incomplete contour map while preserving finer edges.

    Args:
        img: Grayscale or binary image with incomplete contour segments.

    Returns:
        refined_contour: The refined and stable contour.
        contour_img: Image with the refined contour drawn.
    """
    # Ensure input is uint8
    img_uint8 = img.astype(np.uint8)

    # Apply Gaussian Blur to smooth noise while keeping edges
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)

    # Adaptive thresholding to dynamically segment the image
    binary_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological closing (fills small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour by area
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        return None, img  # No contour detected

    return largest_contour

if __name__ == '__main__':
    # Initialize ROS Node
    rospy.init_node('digit_joystick_publisher', anonymous=True)
    pub = rospy.Publisher('digit_joystick', Joy, queue_size=10)
    rospy.Subscriber("emg_raw", Joy, emg_raw_callback)
    rospy.Subscriber("emg_label", Joy, emg_label_callback)
    CCI_max = {"thumb": 0.0, "index": 0.0, "mid": 0.0, "ring": 0.0}
    try:
        record_ind = 0
        count = 0
        ch_tri = 0
        ch_base = 0

        baseline_ring = []

        prev_frames = {
            "thumb": digit0.get_frame(),
            "index": digit1.get_frame(),
            "mid": digit2.get_frame(),
            "ring": digit3.get_frame(),
        }
        current_frames = None

        # Create the trial folder and subfolders
        # folders = create_data_directories()
        while 1:
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


            if count >= 100 and count < 200 and ch_base < 1:
                pass
            if count >= 200 and ch_base < 1:
                ch_base = 1
                print("base mean once")

            frame0 = digit0.get_frame()
            frame1 = digit1.get_frame()
            frame2 = digit2.get_frame()
            frame3 = digit3.get_frame()

            # Compute tactile changes
            pressure_thumb, mask0, CCI0, P_max0, maxLoc0 = estimate_pressure_sobel_filtered(prev_frames["thumb"], frame0)
            pressure_index, mask1, CCI1, P_max1, maxLoc1 = estimate_pressure_sobel_filtered(prev_frames["index"], frame1)
            pressure_mid, mask2, CCI2, P_max2, maxLoc2 = estimate_pressure_sobel_filtered(prev_frames["mid"], frame2)
            pressure_ring, mask3, CCI3, P_max3, maxLoc3 = estimate_pressure_sobel_filtered(prev_frames["ring"], frame3)

            if P_max0 > CCI_max["thumb"]:
                CCI_max["thumb"] = P_max0
            if P_max1 > CCI_max["index"]:
                CCI_max["index"] = P_max1
            if P_max2 > CCI_max["mid"]:
                CCI_max["mid"] = P_max2
            if P_max3 > CCI_max["ring"]:
                CCI_max["ring"] = P_max3

            print(CCI_max)

            mask0 = cv2.flip(mask0, 1)
            frame0 = cv2.flip(frame0, 1)
            smoothed_contour0 = estimate_fine_largest_contour(mask0)

            mask1 = cv2.flip(mask1, 1)
            frame1 = cv2.flip(frame1, 1)
            smoothed_contour1 = estimate_fine_largest_contour(mask1)

            mask2 = cv2.flip(mask2, 1)
            frame2 = cv2.flip(frame2, 1)
            smoothed_contour2 = estimate_fine_largest_contour(mask2)

            mask3 = cv2.flip(mask3, 1)
            frame3 = cv2.flip(frame3, 1)
            smoothed_contour3 = estimate_fine_largest_contour(mask3)

            # Compute convex hull outer
            outer_hull0 = cv2.convexHull(smoothed_contour0)
            convex_area0 = cv2.contourArea(outer_hull0)
            original_area0 = cv2.contourArea(smoothed_contour0)

            outer_hull1 = cv2.convexHull(smoothed_contour1)
            convex_area1 = cv2.contourArea(outer_hull1)
            original_area1 = cv2.contourArea(smoothed_contour1)

            outer_hull2 = cv2.convexHull(smoothed_contour2)
            convex_area2 = cv2.contourArea(outer_hull2)
            original_area2 = cv2.contourArea(smoothed_contour2)

            outer_hull3 = cv2.convexHull(smoothed_contour3)
            convex_area3 = cv2.contourArea(outer_hull3)
            original_area3 = cv2.contourArea(smoothed_contour3)

            # Display tactile images
            frame0_titled = add_title(frame0, 'Right Thumb', (10, 30))
            if (convex_area0 > minimun_hull0):
                cv2.drawContours(frame0_titled, [outer_hull0], -1, (255), thickness=2)
                cv2.drawContours(mask0, [outer_hull0], -1, (255), thickness=2)

            frame1_titled = add_title(frame1, 'Right Index', (10, 30))
            # cv2.drawContours(frame1_titled, [inner_hull1], -1, (255), thickness=2)
            if (convex_area1 > minimun_hull1):
                cv2.drawContours(frame1_titled, [outer_hull1], -1, (255), thickness=2)
                cv2.drawContours(mask1, [outer_hull1], -1, (255), thickness=2)

            frame2_titled = add_title(frame2, 'Right Mid', (10, 30))
            # cv2.drawContours(frame2_titled, [inner_hull2], -1, (255), thickness=2)
            if (convex_area2 > minimun_hull2):
                cv2.drawContours(frame2_titled, [outer_hull2], -1, (255), thickness=2)
                cv2.drawContours(mask2, [outer_hull2], -1, (255), thickness=2)

            frame3_titled = add_title(frame3, 'Right Ring', (10, 30))
            # cv2.drawContours(frame3_titled, [inner_hull3], -1, (255), thickness=2)
            if (convex_area3 > minimun_hull3):
                cv2.drawContours(frame3_titled, [outer_hull3], -1, (255), thickness=2)
                cv2.drawContours(mask3, [outer_hull3], -1, (255), thickness=2)

            # Assume 'img' is the image where you want to draw the circle.
            # For example, to draw a green circle with radius 10 around the maximum point:
            img_with_circle0 = cv2.circle(frame0_titled.copy(), maxLoc0, radius=10, color=(0, 255, 0), thickness=2)
            img_with_circle1 = cv2.circle(frame1_titled.copy(), maxLoc1, radius=10, color=(0, 255, 0), thickness=2)
            img_with_circle2 = cv2.circle(frame2_titled.copy(), maxLoc2, radius=10, color=(0, 255, 0), thickness=2)
            img_with_circle3 = cv2.circle(frame3_titled.copy(), maxLoc3, radius=10, color=(0, 255, 0), thickness=2)

            mask_with_circle0 = cv2.circle(mask0.copy(), maxLoc0, radius=10, color=(0, 255, 0), thickness=2)
            mask_with_circle1 = cv2.circle(mask1.copy(), maxLoc1, radius=10, color=(0, 255, 0), thickness=2)
            mask_with_circle2 = cv2.circle(mask2.copy(), maxLoc2, radius=10, color=(0, 255, 0), thickness=2)
            mask_with_circle3 = cv2.circle(mask3.copy(), maxLoc3, radius=10, color=(0, 255, 0), thickness=2)

            top_row = cv2.hconcat([img_with_circle0, img_with_circle1])
            bottom_row = cv2.hconcat([img_with_circle2, img_with_circle3])
            combined_frame = cv2.vconcat([top_row, bottom_row])

            gap_mask_h = np.ones((320, 50))
            gap_mask_v = np.ones((50, 240))
            gap_mask_m = np.ones((50, 50))

            top_row_mask = cv2.hconcat([mask_with_circle0, gap_mask_h, mask_with_circle1])
            mid_row_mask = cv2.hconcat([gap_mask_v, gap_mask_m, gap_mask_v])
            bottom_row_mask = cv2.hconcat([mask_with_circle2, gap_mask_h, mask_with_circle3])
            combined_frame_mask = cv2.vconcat([top_row_mask, mid_row_mask, bottom_row_mask])

            cv2.imshow('Tactile View', combined_frame)
            cv2.imshow('Greyscale View', combined_frame_mask)

            if ch_base == 1:
                print(CCI_max)
                # Normalize pressure values (0-1 range)
                pressure_values = np.array([float(convex_area0/float(240*320)),
                                            float(convex_area1/float(240*320)),
                                            float(convex_area2/float(240*320)),
                                            float(convex_area3/float(240*320)),])

                CCI_values = np.array([float(CCI0), float(CCI1),
                                       float(CCI2), float(CCI3)])

                Areas = np.array([float(convex_area0), float(convex_area1),
                                       float(convex_area2), float(convex_area3)])

                # # Save the images
                # # e.g., "trial_20250224_132045/images1/image_a.png"
                # path_img_a = os.path.join(folders["rgb_tactile"], f"combined_frame_{record_ind}.png")
                # cv2.imwrite(path_img_a, combined_frame)
                #
                # path_img_b = os.path.join(folders["grey_tactile"], f"combined_frame_mask_{record_ind}.png")
                # cv2.imwrite(path_img_b, combined_frame_mask)
                #
                # # Save the vectors as .npy files
                # # e.g., "trial_20250224_132045/vectors1/vec1.npy"
                # path_vec1 = os.path.join(folders["EDA"], f"EDA_{record_ind}.npy")
                # np.save(path_vec1, pressure_values)
                #
                # path_vec2 = os.path.join(folders["CCI"], f"CCI_{record_ind}.npy")
                # np.save(path_vec2, CCI_values)
                #
                # path_vec3 = os.path.join(folders["convex_area"], f"convex_area_{record_ind}.npy")
                # np.save(path_vec3, Areas)
                #
                # path_vec4 = os.path.join(folders["emg_raw"], f"emg_raw_{record_ind}.npy")
                # np.save(path_vec4, emg_raw_data)
                #
                # path_vec5 = os.path.join(folders["emg_label"], f"emg_label_{record_ind}.npy")
                # np.save(path_vec5, emg_label_data)

                record_ind += 1


                # Generate joystick message
                joy_msg = Joy()
                joy_msg.header.stamp = rospy.Time.now()

                joy_msg.axes = [0.0] * 8 # Adjust size if needed
                joy_msg.axes[0] = pressure_values[0]  # Thumb pressure → Axis 0
                joy_msg.axes[1] = pressure_values[1]  # Index finger → Axis 1
                joy_msg.axes[2] = pressure_values[2]  # Middle finger → Axis 2
                joy_msg.axes[3] = pressure_values[3]  # Ring finger → Axis 3

                joy_msg.axes[4] = CCI_values[0]  # Thumb pressure → Axis 0
                joy_msg.axes[5] = CCI_values[1]  # Index finger → Axis 1
                joy_msg.axes[6] = CCI_values[2]  # Middle finger → Axis 2
                joy_msg.axes[7] = CCI_values[3]  # Ring finger → Axis 3

                pub.publish(joy_msg)

            if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                break


    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        cv2.destroyAllWindows()