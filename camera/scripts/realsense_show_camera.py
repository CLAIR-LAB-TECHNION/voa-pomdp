import cv2
import pyrealsense2 as rs
import numpy as np

print("t") 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    color_image = np.asanyarray(color_frame.get_data())

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    ret, buffer = cv2.imencode('.jpg', color_image)
    frame = buffer.tobytes()
    cv2.imshow('image', color_image)
    cv2.waitKey(1)

