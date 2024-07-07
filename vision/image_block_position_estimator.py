from vision.utils import crop_workspace
from vision.object_detection import ObjectDetection
import json
from motion_planning.geometry_and_transforms import GeometryAndTransforms
import numpy as np
import matplotlib.pyplot as plt
from camera.realsense_camera import project_color_pixel_to_depth_pixel
from camera.configurations_and_params import color_camera_intrinsic_matrix


class ImageBlockPositionEstimator:
    def __init__(self, workspace_limits_x, workspace_limits_y, gt: GeometryAndTransforms, robot_name='ur5e_1'):
        self.workspace_limits_x = workspace_limits_x
        self.workspace_limits_y = workspace_limits_y
        self.gt = gt
        self.detector = ObjectDetection()
        self.robot_name = robot_name

    def bboxes_cropped_to_orig(self, bboxes, xyxy):
        bboxes_orig = bboxes.cpu().numpy().copy()
        bboxes_orig[:, 0] += xyxy[0]
        bboxes_orig[:, 1] += xyxy[1]
        bboxes_orig[:, 2] += xyxy[0]
        bboxes_orig[:, 3] += xyxy[1]
        return bboxes_orig

    def points_image_to_camera_frame(self, points_image_xy, z_depth):
        """
        Transform points from image coordinates to camera frame coordinates
        :param points_image_xy: points in image coordinates nx2
        :param z_depth: depth of the points nx1
        :return: points in camera frame coordinates nx3
        """
        fx = color_camera_intrinsic_matrix[0, 0]
        fy = color_camera_intrinsic_matrix[1, 1]
        cx = color_camera_intrinsic_matrix[0, 2]
        cy = color_camera_intrinsic_matrix[1, 2]

        x_camera = (points_image_xy[:, 0] - cx) * z_depth / fx
        y_camera = (points_image_xy[:, 1] - cy) * z_depth / fy
        points_camera_frame = np.vstack((x_camera, y_camera, z_depth)).T

        return points_camera_frame

    def get_z_depth_mean(self, points_color_image_xy, depth_image, windows_sizes):
        """
        Get mean depth of the points in the window around the points. discard points with depth 0.
        This method project the point in from image to depth
        :param points_color_image_xy: the point in color image coordinates
        :param depth_image:
        :param windows_sizes:
        :return: mean depth in windows around points, -1 if no depth around the point or invalid points
        """
        points_in_depth = [project_color_pixel_to_depth_pixel(point, depth_image) for point in points_color_image_xy]


        depths = []
        for point, win_size in zip(points_in_depth, windows_sizes):
            if point[0] < 0 or point[1] < 0 or point[0] >= depth_image.shape[1] or point[1] >= depth_image.shape[0]:
                depths.append(-1)
                continue

            # make sure window is within the image
            x_min = max(0, point[0] - win_size[0])
            x_max = min(depth_image.shape[1], point[0] + win_size[0])
            y_min = max(0, point[1] - win_size[1])
            y_max = min(depth_image.shape[0], point[1] + win_size[1])

            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

            window = depth_image[y_min:y_max, x_min:x_max]
            if np.all(window <= 0):
                depths.append(-1)
                continue

            depths.append(np.mean(window[window > 0]))

        return depths


    def get_block_positions_depth(self, images, depths, robot_configurations, z_offset=0.02, detect_on_cropped=True):
        """
        Get block positions from images and depth images. can be single image or batch of images
        :param images: ims to get block positions from
        :param depths: depths to get block positions from
        :param robot_configurations: robot configurations for each image
        :param z_offset: offset from the detected depth to the block position, this is the length of the block,
            since depth is measured to face and we want to get the center of the block
        :param detect_on_cropped: if True, detect objects on cropped images, if False, detect on full images
        :return: list of detected block positions in world coordinates for each image
            (or single list if single image is given)
        """
        if len(np.array(images).shape) == 3:
            images = [images]
            robot_configurations = [robot_configurations]
            depths = [depths]

        if detect_on_cropped:
            cropped_images = []
            cropped_images_xyxy = []
            for image, robot_config in zip(images, robot_configurations):
                cropped_image, xyxy = crop_workspace(image, robot_config, self.gt, self.workspace_limits_x,
                                                     self.workspace_limits_y)
                cropped_images.append(cropped_image)
                cropped_images_xyxy.append(xyxy)

            bboxes_in_cropped, _, results = self.detector.detect_objects(cropped_images)
            bboxes = [self.bboxes_cropped_to_orig(bboxes, xyxy)
                      for bboxes, xyxy in zip(bboxes_in_cropped, cropped_images_xyxy)]
        else:
            bboxes, _, results = self.detector.detect_objects(images)

        bboxes_centers = [(bbox[:, :2] + bbox[:, 2:]) / 2 for bbox in bboxes]
        bbox_sizes = [(bbox[:, 2:] - bbox[:, :2]) for bbox in bboxes]

        estimated_z_depths = []
        for bbox_center, bbox_sizes_curr, depth_im in zip(bboxes_centers, bbox_sizes, depths):
            # take at least 3x3 pixels, at most 10x10 pixels, and as default half of the bbox size,
            # separately for x and y
            bbox_sizes_curr = np.array(bbox_sizes_curr)
            win_sizes_x = np.clip(np.ceil(bbox_sizes_curr[:, 0] / 2), 3, 10)
            win_sizes_y = np.clip(np.ceil(bbox_sizes_curr[:, 1] / 2), 3, 10)
            windows_sizes = np.vstack((win_sizes_x, win_sizes_y)).T

            estimated_z_depths_curr = self.get_z_depth_mean(bbox_center, depth_im, windows_sizes)
            estimated_z_depths_curr = np.array(estimated_z_depths_curr) + z_offset
            estimated_z_depths.append(estimated_z_depths_curr)

        # TODO: plot bbox, center, center in depth

        block_positions_camera_frame = []
        for bbox_center, depth in zip(bboxes_centers, estimated_z_depths):
            block_positions_camera_frame.append(self.points_image_to_camera_frame(bbox_center, depth))

        block_positions_world = []
        for block_positions_camera_frame_curr, robot_config in zip(block_positions_camera_frame, robot_configurations):
            # we do point by point, this method is not validated as vectorized yet, need to do that
            curr_block_positions_world = []
            for bpos_cam in block_positions_camera_frame_curr:
                block_position_world = self.gt.point_camera_to_world(bpos_cam, self.robot_name, robot_config)
                curr_block_positions_world.append(block_position_world)
            block_positions_world.append(curr_block_positions_world)
        # if single image, return single list
        if len(block_positions_world) == 1:
            return block_positions_world[0]
        return block_positions_world
