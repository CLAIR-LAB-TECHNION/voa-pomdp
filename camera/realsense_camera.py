import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt


class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * self.depth_scale
        return color_image, depth_image

    # TODO: Visualize depth utility, visualize both, align

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
    camera = RealsenseCamera()
    max_depth = 5  # actual is 16
    while True:
        rgb, depth = camera.get_frame()
        depth = np.clip(depth, 0, max_depth)
        if rgb is not None and depth is not None:
            # scale just for cv2:
            depth = depth / max_depth
            depth = (depth * 255).astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            cv2.imshow('image', rgb)
            cv2.imshow('depth', depth)
            cv2.waitKey(1)
