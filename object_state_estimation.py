import os

import numpy as np
from matplotlib import pyplot as plt

from vision.object_detection import ObjectDetection

detector = ObjectDetection()


image_indices = list(range(3, 4))
loaded_depths = []
loaded_images = []
for idx in image_indices:
    image_path = os.path.join("images_data_merged_hires/images", f'image_{idx}.npy')
    if os.path.exists(image_path):
        image_array = np.load(image_path)
        loaded_images.append(image_array)
    else:
        print(f"Image {image_path} does not exist.")

    depth_path = os.path.join("images_data_merged_hires/depth", f'depth_{idx}.npy')
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
depth_clipped = np.clip(loaded_depths[0], 0, 4)
plt.imshow(depth_clipped)
plt.scatter(boxes_center[:, 0], boxes_center[:, 1], c='g', s=5)
plt.show()

# TODO: no depth above 0.5 m. use plane projection, and later figure out if 0.5m is close enough
#  one option may be to crop image to region of interest and then call od?

pass


