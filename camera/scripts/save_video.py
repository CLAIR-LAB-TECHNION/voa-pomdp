import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import os

# Function to generate frame by frame from camera
def gen_frames():  
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
	pipeline.start(config)


	while True:
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()

		if not color_frame:
			continue


		color_image = np.asanyarray(color_frame.get_data())
		
		timestamp = str(datetime.now()).split(' ')[-1]
		
		cv2.imwrite(f'vid/img_{timestamp}.png', color_image)


if __name__ == '__main__':
	os.system('rm vid/*')
	gen_frames()

