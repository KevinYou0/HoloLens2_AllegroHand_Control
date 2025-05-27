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
from scipy.stats import chi2
ESC_KEY = 27

# DIGIT Sensors Settings
digits = DigitHandler.list_digits()
pprint.pprint(digits)

# Connect to DIGIT devices
digit3 = Digit("D21068", "Right Ring")

for digit in [digit3]:
    digit.connect()
    digit.set_intensity(Digit.LIGHTING_MAX)
    digit.set_resolution(Digit.STREAMS["QVGA"])
    digit.set_fps(Digit.STREAMS["QVGA"]["fps"]["30fps"])

# Function to add a title to an image
def add_title(img, title, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA):
    cv2.putText(img, title, pos, font, font_scale, font_color, thickness, line_type)
    return img


def min_max_normalize(grayscale_img):
    """
    Perform Min-Max normalization on a grayscale tactile image.

    Parameters:
    - grayscale_img: 2D numpy array (grayscale tactile image)

    Returns:
    - normalized_img: Min-max normalized image (range [0,1])
    """
    # Convert to float for numerical stability
    grayscale_img = grayscale_img.astype(np.float32)

    # Compute min and max values
    I_min = np.min(grayscale_img)
    I_max = np.max(grayscale_img)

    # Avoid division by zero
    if I_max - I_min == 0:
        return np.zeros_like(grayscale_img)

    # Apply Min-Max normalization
    normalized_img = 255 * (grayscale_img - I_min) / (I_max - I_min)
    return normalized_img

def preprocess_frame_bilateral(frame, d=15, sigmaColor=150, sigmaSpace=150):
    """
    Applies bilateral filtering to reduce high-frequency noise while preserving edges.

    Args:
        frame: Raw tactile image (RGB).

    Returns:
        preprocessed_frame: Noise-reduced grayscale image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter (d = 9, sigmaColor = 75, sigmaSpace = 75)
    gray_filtered = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

    return gray_filtered

def overlay_EDA_mask(grayscale_img, EDA_mask):
    """
    Overlay the EDA mask onto the grayscale tactile image.
    The mask is shown in red.
    """
    overlay = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for coloring
    overlay[EDA_mask] = [0, 0, 255]  # Set masked regions to red

    return overlay


def compute_CCI_EDA(grayscale_img, confidence_level):
    """
    Compute Contact Concentration Index (CCI) and Effective Deformation Area (EDA)
    from a grayscale tactile image.

    Parameters:
    - grayscale_img: 2D numpy array (grayscale tactile image)
    - confidence_level: percentage of high-pressure pixels included in EDA computation

    Returns:
    - CCI (Contact Concentration Index)
    - EDA (Effective Deformation Area) in pixel count
    """

    # Normalize image to range [0, 1] as pseudo-pressure map
    pseudo_pressure = grayscale_img.astype(np.float32) / 255.0

    # Compute threshold based on top `percentile`% pixel values
    threshold_value = np.percentile(pseudo_pressure, confidence_level)

    # Generate EDA mask by selecting top `percentile`% pressure values
    EDA_mask = pseudo_pressure >= threshold_value

    # Compute Effective Deformation Area (EDA) as the number of pixels above the threshold
    EDA = np.sum(EDA_mask)

    # Get max pressure value
    P_max = np.max(pseudo_pressure)

    # Compute Contact Concentration Index (CCI)
    ECA = EDA  # ECA is now directly the count of threshold-exceeding pixels
    CCI = P_max / (ECA if ECA > 0 else 1)  # Avoid division by zero

    return CCI, EDA, EDA_mask


def detect_edges_canny(grayscale_img, low_threshold=50, high_threshold=150):
    """
    Detect edges in a grayscale tactile image using Canny edge detection.

    Parameters:
    - grayscale_img: 2D numpy array (grayscale tactile image)
    - low_threshold: Lower threshold for Canny edge detection
    - high_threshold: Higher threshold for Canny edge detection

    Returns:
    - edge_mask: Binary mask showing detected edges
    """
    # Apply Gaussian Blur to reduce noise
    blurred_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)

    # Perform Canny edge detection
    edge_mask = cv2.Canny(blurred_img, low_threshold, high_threshold)

    return edge_mask

def detect_shadows(frame, blockSize=21, C=10):
    """
    Detect shadows in a tactile image using adaptive thresholding.

    Args:
        frame: Raw tactile image (RGB or grayscale).
        blockSize: Size of pixel neighborhood used for thresholding (odd integer).
        C: Constant subtracted from the mean intensity (controls threshold).

    Returns:
        shadow_mask: Binary mask highlighting shadow regions.
    """
    # Convert to grayscale if necessary
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Adaptive thresholding to find shadows
    shadow_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, blockSize, C)

    # Morphological operations to remove noise and refine shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    return shadow_mask

def detect_edges_sobel(grayscale_img):
    """
    Detect edges using the Sobel operator.

    Parameters:
    - grayscale_img: 2D numpy array (grayscale tactile image)

    Returns:
    - edge_mask: Binary mask showing detected edges
    """
    # Compute gradients in X and Y directions
    sobel_x = cv2.Sobel(grayscale_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(grayscale_img, cv2.CV_64F, 0, 1, ksize=5)

    # Compute gradient magnitude
    edge_mask = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize to 8-bit format
    edge_mask = cv2.normalize(edge_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return edge_mask

base_ring = None
if __name__ == '__main__':
    try:
        count = 0
        ch_tri = 0
        ch_base = 0

        baseline_ring = []

        prev_frames = {
            "ring": digit3.get_frame(),
        }
        current_frames = None

        while not rospy.is_shutdown():
            count += 1
            if count >= 100 and ch_tri < 1:
                prev_frames = {
                    "ring": digit3.get_frame(),
                }
                ch_tri = 1
                print("select Once")

            if count >= 1000:
                count = 0

            if count >= 200 and ch_base < 1:
                ch_base = 1
                base_ring = digit3.get_frame()
                print("base mean once")

            if base_ring is not None:
                frame3 = digit3.get_frame()
                frame3_norm = np.absolute(frame3 - base_ring)
                # Compute tactile changes
                grayscale_img = preprocess_frame_bilateral(frame3_norm, 15, 150, 150)
                grayscale_img_norm = min_max_normalize(grayscale_img)

                shadow_frame = detect_shadows(grayscale_img)
                sobel_edges = detect_edges_sobel(grayscale_img_norm)

                CCI, EDA, EDA_mask = compute_CCI_EDA(grayscale_img, 20)
                EDA_visual = overlay_EDA_mask(grayscale_img, EDA_mask)

                cv2.imshow('Normal View', frame3_norm)
                cv2.imshow('Shadow View', shadow_frame)
                cv2.imshow('Sobel View', sobel_edges)

            if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                break

    finally:
        cv2.destroyAllWindows()