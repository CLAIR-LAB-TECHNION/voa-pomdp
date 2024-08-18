import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
from lab_ur_stack.camera.configurations_and_params import depth_fx, depth_fy, depth_ppx, depth_ppy, color_fx, color_fy, color_ppx, \
    color_ppy, depth_to_color_translation
from lab_ur_stack.camera.utils import get_mean_depth


def project_color_pixel_to_depth_pixel(color_image_point, depth_image):
    # the projection depends on depth of the point which we don't know yet.
    # we will take the mean depth of the region around the pixel without projection

    mean_depth = get_mean_depth(depth_image, color_image_point, window_size=20)
    if mean_depth == -1:
        return (-1, -1)  # no depth around that point, cannot project

    x_color = (color_image_point[0] - color_ppx) / color_fx
    y_color = (color_image_point[1] - color_ppy) / color_fy

    color_frame_point = np.array([x_color * mean_depth, y_color * mean_depth, mean_depth])

    depth_frame_point = color_frame_point - depth_to_color_translation

    depth_image_point = [(depth_frame_point[0] * depth_fx / depth_frame_point[2]) + depth_ppx,
                         (depth_frame_point[1] * depth_fy / depth_frame_point[2]) + depth_ppy]

    return depth_image_point


class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        self.pipeline.start(self.config)

        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_frame_bgr(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * self.depth_scale
        return color_image, depth_image

    def get_frame_rgb(self):
        bgr, depth = self.get_frame_bgr()
        rgb = None
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, depth

    def plotable_depth(self, depth_image, max_depth=3):
        depth_image = np.clip(depth_image, 0, max_depth)
        depth_image = (depth_image / max_depth * 255).astype(np.uint8)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        return depth_image

    def plot_depth(self, depth_image, max_depth=3):
        plotable_depth = self.plotable_depth(depth_image, max_depth)
        plt.imshow(plotable_depth)
        plt.show()

    def plot_rgb(self, rgb_image):
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()


if __name__ == "__main__":
    pass
