import os
import cv2
import time
import numpy as np
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils_orbec import frame_to_bgr_image
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler
import pprint

SAVE_FOLDER = "captured_frames"  # Folder to save images
FPS = 5  # Save 5 frames per second
ESC_KEY = 27

# DIGITs Sensor Settings
digits = DigitHandler.list_digits()
pprint.pprint(digits)

# Connect to DIGIT devices
digit0 = Digit("D21063", "Right Thumb")
digit0.connect()
digit1 = Digit("D21064", "Right Index")
digit1.connect()
digit2 = Digit("D21066", "Right Mid")
digit2.connect()

# Set DIGIT settings (resolution and FPS)
digit0.set_intensity(Digit.LIGHTING_MAX)
digit1.set_intensity(Digit.LIGHTING_MAX)
digit2.set_intensity(Digit.LIGHTING_MAX)

qvga_res = Digit.STREAMS["QVGA"]
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]

digit0.set_resolution(qvga_res)
digit1.set_resolution(qvga_res)
digit2.set_resolution(qvga_res)

digit0.set_fps(fps_30)
digit1.set_fps(fps_30)
digit2.set_fps(fps_30)


# Function to add a title to an image
def add_title(img, title, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), thickness=2,
              line_type=cv2.LINE_AA):
    cv2.putText(img, title, pos, font, font_scale, font_color, thickness, line_type)
    return img


def depth_to_color(depth_frame):
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
    normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    config = Config()
    pipeline = Pipeline()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError:
            color_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(color_profile)

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

    except Exception as e:
        print(f"Error setting up pipeline: {e}")
        return

    pipeline.start(config)

    current_minute = 1
    frames_per_minute = FPS * 60

    try:
        while True:
            minute_folder = os.path.join(SAVE_FOLDER, f"minute{current_minute}")
            os.makedirs(os.path.join(minute_folder, "frame0"), exist_ok=True)
            os.makedirs(os.path.join(minute_folder, "frame1"), exist_ok=True)
            os.makedirs(os.path.join(minute_folder, "frame2"), exist_ok=True)
            os.makedirs(os.path.join(minute_folder, "depth_viewer"), exist_ok=True)
            os.makedirs(os.path.join(minute_folder, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(minute_folder, "npy_files"), exist_ok=True)

            start_time = time.time()
            for frame_count in range(frames_per_minute):
                loop_start = time.time()

                frames = pipeline.wait_for_frames(100)
                if frames:
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if color_frame and depth_frame:
                        color_image = frame_to_bgr_image(color_frame)
                        depth_image = depth_to_color(depth_frame)

                        if color_image is not None and depth_image is not None:
                            frame0 = digit0.get_frame()
                            frame1 = digit1.get_frame()
                            frame2 = digit2.get_frame()

                            frame0_titled = add_title(frame0, 'Right Thumb', (10, 30))
                            frame1_titled = add_title(frame1, 'Right Index', (10, 30))
                            frame2_titled = add_title(frame2, 'Right Mid', (10, 30))

                            frame0_path = os.path.join(minute_folder, "frame0", f"frame0_{frame_count}.jpg")
                            frame1_path = os.path.join(minute_folder, "frame1", f"frame1_{frame_count}.jpg")
                            frame2_path = os.path.join(minute_folder, "frame2", f"frame2_{frame_count}.jpg")
                            depth_viewer_path = os.path.join(minute_folder, "depth_viewer",
                                                             f"depth_viewer_{frame_count}.jpg")
                            rgb_path = os.path.join(minute_folder, "rgb", f"rgb_{frame_count}.jpg")
                            npy_file_path = os.path.join(minute_folder, "npy_files", f"depth_{frame_count}.npy")

                            cv2.imwrite(frame0_path, frame0_titled)
                            cv2.imwrite(frame1_path, frame1_titled)
                            cv2.imwrite(frame2_path, frame2_titled)
                            cv2.imwrite(depth_viewer_path, depth_image)
                            cv2.imwrite(rgb_path, color_image)

                            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                            np.save(npy_file_path, depth_data)

                            # Calculate sleep time to maintain FPS
                            elapsed = time.time() - loop_start
                            sleep_time = max(0, (1 / FPS) - elapsed)
                            time.sleep(sleep_time)
                    else:
                        print(f"Missing color or depth frame at minute {current_minute}, frame {frame_count}")
                else:
                    print(f"No frames received at minute {current_minute}, frame {frame_count}")

                if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                    return

            end_time = time.time()
            print(
                f"Minute {current_minute} completed. Captured {frame_count + 1} frames in {end_time - start_time:.2f} seconds")
            current_minute += 1

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
