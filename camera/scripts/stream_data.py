from flask import Flask, Response
import pyrealsense2 as rs
import numpy as np

app = Flask(__name__)

def get_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def generate_depth_stream():
    pipeline = get_pipeline()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            yield depth_image.tobytes()
    finally:
        pipeline.stop()

def generate_color_stream():
    pipeline = get_pipeline()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            yield color_image.tobytes()
    finally:
        pipeline.stop()

@app.route('/depth_feed')
def depth_feed():
    return Response(generate_depth_stream(), mimetype='application/octet-stream')

@app.route('/color_feed')
def color_feed():
    return Response(generate_color_stream(), mimetype='application/octet-stream')

if __name__ == '__main__':
    app.run(host='192.168.0.3', port=5000, threaded=True)

