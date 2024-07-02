import os

import numpy as np

from vision.object_detection import ObjectDetection

detector = ObjectDetection()


image_indices = list(range(1, 2))
loaded_images = []
for idx in image_indices:
    image_path = os.path.join("images_data_merged_hires/images", f'image_{idx}.npy')
    if os.path.exists(image_path):
        image_array = np.load(image_path)
        loaded_images.append(image_array)
    else:
        print(f"Image {image_path} does not exist.")

bboxes, _, _ = detector.detect_objects(loaded_images[0])

# work with one image for now:
bboxes = bboxes[0]



