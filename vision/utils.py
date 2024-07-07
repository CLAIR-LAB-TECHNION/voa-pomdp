import json

from matplotlib import pyplot as plt

from motion_planning.geometry_and_transforms import GeometryAndTransforms
import numpy as np
from camera.configurations_and_params import color_camera_intrinsic_matrix


def project_points_to_image(points, gt: GeometryAndTransforms, robot_name, robot_config):
    """
    Project points from the world coordinates to the image coordinates
    :param points: points to project nx3 array
    :param gt: geometric transforms object
    :param robot_name: robot with the camera
    :param robot_config: configuration of the robot with the camera
    :return: projected points in the image coordinates nx2 array
    """
    points = np.array(points)
    if points.shape == (3,):  # only one point
        points = points.reshape(1, 3)

    assert points.shape[1] == 3, "points should be nx3 array"

    world_2_camera = gt.world_to_camera_transform(robot_name, robot_config)
    world_2_camera = gt.se3_to_4x4(world_2_camera)

    points_homogenous = np.ones((4, points.shape[0]))
    points_homogenous[:3] = points.T

    # points are column vectors as required
    points_camera_frame_homogenous = world_2_camera @ points_homogenous  # 4x4 @ 4xn = 4xn
    points_camera_frame = points_camera_frame_homogenous[:3]
    points_image_homogenous = color_camera_intrinsic_matrix @ points_camera_frame  # 3x3 @ 3xn = 3xn
    points_image = points_image_homogenous / points_image_homogenous[2]  # normalize

    return points_image[:2].T


def crop_workspace(image,
                   robot_config,
                   gt: GeometryAndTransforms,
                   workspace_limits_x,
                   workspace_limits_y,
                   z=-0.0,
                   robot_name="ur5e_1",
                   extension_radius=0.04,):
    """
    crop the workspace that is within given workspace limits and return the cropped image and coordinates of
     the cropped image in the original image
    :param image: the image to crop
    :param robot_config: configuration of the robot with the camera, ur5e_1 if not specified
    :param gt: geometric transforms object
    :param workspace_limits_x: workspace max and min limits in x direction in world coordinates
    :param workspace_limits_y: workspace max and min limits in y direction in world coordinates
    :param robot_name: robot with the camera
    :param extension_radius: how much to extend the workspace. half of the box can be outside the workspace limits, thus
     we need to extend at least by half box which is 2cm, but due to noice and inaccuracies, we extend more as default
    :return: cropped image, xyxy within the original image
    """
    extended_x_lim = [workspace_limits_x[0] - extension_radius, workspace_limits_x[1] + extension_radius]
    extended_y_lim = [workspace_limits_y[0] - extension_radius, workspace_limits_y[1] + extension_radius]
    z_lim = [z - extension_radius, z + extension_radius]

    corners = np.array([[extended_x_lim[0], extended_y_lim[0], z_lim[0]],
                        [extended_x_lim[0], extended_y_lim[0], z_lim[1]],
                        [extended_x_lim[0], extended_y_lim[1], z_lim[0]],
                        [extended_x_lim[0], extended_y_lim[1], z_lim[1]],
                        [extended_x_lim[1], extended_y_lim[0], z_lim[0]],
                        [extended_x_lim[1], extended_y_lim[0], z_lim[1]],
                        [extended_x_lim[1], extended_y_lim[1], z_lim[0]],
                        [extended_x_lim[1], extended_y_lim[1], z_lim[1]]])

    corners_image = project_points_to_image(corners, gt, robot_name, robot_config)
    corners_image = corners_image.astype(int)
    x_min, y_min = np.min(corners_image, axis=0)
    x_max, y_max = np.max(corners_image, axis=0)

    # in case the workspace is out of the image:
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, [x_min, y_min, x_max, y_max]


def sample_sensor_configs(workspace_limits_x, workspace_limits_y, z=-0.0, num_samples=10):
    pass


if __name__ == "__main__":
    pass
