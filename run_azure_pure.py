import cv2
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A
def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510
    count = 0
    while 1:
        capture = k4a.get_capture()
        if np.any(capture.color):
            cv2.imshow("k4a", capture.color[:, :, :3])
            save_path = "/home/hengxuy/pipe_new_folder/pipe_image_color/" + str(count) + ".jpg"
            cv2.imwrite(save_path, capture.color)  
        
        if np.any(capture.depth):
            save_path = "/home/hengxuy/pipe_new_folder/pipe_depth/" + str(count) + ".npy"
            np.save(save_path, capture.depth)
        if np.any(capture.depth_point_cloud):
            save_path = "/home/hengxuy/pipe_new_folder/pc_pts/" + str(count) + ".npy"
            np.save(save_path, capture.depth_point_cloud.reshape((-1, 3)))
            save_path = "/home/hengxuy/pipe_new_folder/pc_color/" + str(count) + ".npy"
            np.save(save_path, capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)))

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break
        count += 1
    k4a.stop()


if __name__ == "__main__":
    main()