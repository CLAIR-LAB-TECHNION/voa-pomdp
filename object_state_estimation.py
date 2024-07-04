import os
import numpy as np
from matplotlib import pyplot as plt
from vision.object_detection import ObjectDetection
import json


def load_camera_params(filename):
    with open(filename, "r") as f:
        params = json.load(f)
    return params


def get_mean_depth(depth_image, color_pixel, window_size=10):
    x, y = color_pixel
    x, y = int(x), int(y)
    half_window = window_size // 2

    # Define the region of interest (ROI)
    roi = depth_image[max(0, y-half_window):min(depth_image.shape[0], y+half_window),
                      max(0, x-half_window):min(depth_image.shape[1], x+half_window)]

    # Get valid depth values (non-zero)
    valid_depths = roi[roi > 0]

    if len(valid_depths) == 0:
        return -1

    mean_depth = np.mean(valid_depths)
    return mean_depth

def project_color_pixel_to_depth_pixel(color_pixel, depth_image, camera_params):
    # Load intrinsic and extrinsic parameters
    depth_intr = camera_params['depth_intrinsics']
    color_intr = camera_params['color_intrinsics']
    depth_to_color_extr = camera_params['depth_to_color_extrinsics']

    # Convert intrinsics to numpy arrays
    depth_fx = depth_intr['fx']
    depth_fy = depth_intr['fy']
    depth_ppx = depth_intr['ppx']
    depth_ppy = depth_intr['ppy']

    color_fx = color_intr['fx']
    color_fy = color_intr['fy']
    color_ppx = color_intr['ppx']
    color_ppy = color_intr['ppy']

    translation = np.array(depth_to_color_extr['translation'])

    # Get the mean depth value around the color pixel
    mean_depth = get_mean_depth(depth_image, color_pixel)

    if mean_depth == -1:
        return (-1, -1)

    # Convert the color pixel to normalized coordinates in color space
    x_color = (color_pixel[0] - color_ppx) / color_fx
    y_color = (color_pixel[1] - color_ppy) / color_fy

    # Convert to 3D point in color camera space
    color_point = np.array([x_color * mean_depth, y_color * mean_depth, mean_depth])

    # Transform the 3D point from color camera space to depth camera space
    depth_point = color_point - translation

    # Project the depth camera 3D point to 2D depth image pixel coordinates
    u_depth = int((depth_point[0] * depth_fx / depth_point[2]) + depth_ppx)
    v_depth = int((depth_point[1] * depth_fy / depth_point[2]) + depth_ppy)

    return (u_depth, v_depth)


# Load camera parameters
camera_params = load_camera_params("camera_params.json")

detector = ObjectDetection()


image_indices = list(range(8, 8))
loaded_depths = []
loaded_images = []
for idx in image_indices:
    image_path = os.path.join("vision/images_data_merged_hires/images", f'image_{idx}.npy')
    if os.path.exists(image_path):
        image_array = np.load(image_path)
        loaded_images.append(image_array)
    else:
        print(f"Image {image_path} does not exist.")

    depth_path = os.path.join("vision/images_data_merged_hires/depth", f'depth_{idx}.npy')
    if os.path.exists(depth_path):
        depth_array = np.load(depth_path)
        loaded_depths.append(depth_array)
    else:
        print(f"Depth {depth_path} does not exist.")


bboxes, _, results = detector.detect_objects(loaded_images)

# work with one image for now:
results = results[0]
bboxes = bboxes[0].cpu()

plt.imshow(results.plot())

boxes_center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
# add boxes center to plot:
plt.scatter(boxes_center[:, 0], boxes_center[:, 1], c='g', s=5)
plt.show()

# plot centers also on depth image:
centers_depth = [project_color_pixel_to_depth_pixel(center, loaded_depths[0], camera_params) for center in boxes_center]
centers_depth = np.array(centers_depth)

depth_clipped = np.clip(loaded_depths[0], 0, 3)
plt.imshow(depth_clipped, cmap='gray')
plt.scatter(centers_depth[:, 0], centers_depth[:, 1], c='g', s=5)
plt.show()

# TODO: no depth above 0.5 m. use plane projection, and later figure out if 0.5m is close enough
#  one option may be to crop image to region of interest and then call od?

pass


