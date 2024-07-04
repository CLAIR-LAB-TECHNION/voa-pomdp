from camera.realsense_camera import RealsenseCamera
import time
import os
import cv2

cam = RealsenseCamera()

time.sleep(5)

for i in range(20):
    time.sleep(1)
    im, _ = cam.get_frame()
    cv2.imwrite(f"calibration_images/image_{i}.png", im)
    print(f"Saved image {i}")
