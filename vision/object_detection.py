from matplotlib import pyplot as plt
from ultralytics import YOLOWorld
import cv2
import numpy as np
import os


default_classes = (
    'wooden box', 'box', 'top view of wooden box',
    'wooden block', 'block', 'top view of wooden block',
    'wooden cube', 'cube', 'top view of wooden cube',
)

class ObjectDetection:
    def __init__(self, classes=default_classes, min_confidence=0.05):
        self.min_confidence = min_confidence

        self.yolo = YOLOWorld('yolov8x-worldv2')  # largest model
        self.yolo.set_classes(classes)

    def detect_objects(self, im_arr, ):
        '''

        :param im_arr: b x w x h x 3 numpy array
        :return: a list of tuples for each object detected in the image. The tuple is bboxes, confidences, result object
        '''
        results = self.yolo.predict(im_arr, conf=self.min_confidence)

        ret = []
        for r in results:
            bboxes = r.boxes.xyxy
            confidences = r.boxes.conf
            ret.append((bboxes, confidences, r))

        return ret

    def get_annotated_images(self, result):
        return result.plot()


if __name__ == "__main__":
    detector = ObjectDetection()

    image_indices = list(range(1, 35))
    loaded_images = []
    for idx in image_indices:
        image_path = os.path.join("images_data_merged/images", f'image_{idx}.npy')
        if os.path.exists(image_path):
            image_array = np.load(image_path)
            loaded_images.append(image_array)
        else:
            print(f"Image {image_path} does not exist.")

    for r in detector.detect_objects(loaded_images):
        im_annotated = detector.get_annotated_images(r[2])
        plt.imshow(im_annotated)
        plt.show()
        ##### TODO: better methods for different types of outputs
