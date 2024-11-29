import numpy as np
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms


mujoco_camera_intrinsics = np.array([[869.12, 0, 640],
                                     [0, 869.12, 360],
                                     [0, 0, 1]])


def in_fov(points3d, extrinsic_matrix, cam_intrinsic_matrix, image_size, margin_in_pixels):
    """
    calculate if a list of 3d points are in the camera's field of view
    @param points3d: list of 3d points
    @param extrinsic_matrix: camera extrinsic matrix
    @param cam_intrinsic_matrix: camera intrinsic matrix
    @param image_size: image size
    @param margin_in_pixels: if a point is in image but close to the edge by this margin in one of the axes,
     it is considered out of fov
    @return: list of True/False for each point whether it is in the camera's field of view
    """
    if points3d.shape == (3,):
        points3d = np.array([points3d])

    points3d = np.asarray(points3d)
    points3d = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=1)  # n x 4
    points3d = points3d.T  # 4 x n

    points3d_cam = extrinsic_matrix @ points3d  # 4 x n
    points3d_cam_non_homogeneous = points3d_cam[:3, :]  # 3 x n
    points2d = cam_intrinsic_matrix @ points3d_cam_non_homogeneous  # 3 x n
    points2d = points2d / points2d[2, :]

    x = points2d[0, :]
    y = points2d[1, :]

    min_x, min_y = margin_in_pixels, margin_in_pixels
    max_x, max_y = image_size[0] - margin_in_pixels, image_size[1] - margin_in_pixels
    in_fov = (min_x <= x) & (x <= max_x) & (min_y <= y) & (y <= max_y)

    return in_fov


def in_r1_fov(points3d, config, gt: GeometryAndTransforms, cam_intrinsic_matrix, image_size=(1280, 720),
              margin_in_pixels=30):
    """
    calculate if a list of 3d points are in the camera's field of view
    @param points3d: list of 3d points
    @param config: robot config
    @param gt: geometry and transforms object
    @param cam_intrinsic_matrix: camera intrinsic matrix
        @param image_size: image size
    @param margin_in_pixels: if a point is in image but close to the edge by this margin in one of the axes,
     it is considered out of fov
    @return: list of True/False for each point whether it is in the camera's field of view
    """
    extrinsic_transform = gt.world_to_camera_transform("ur5e_1", config)
    extrinsic_matrix = gt.se3_to_4x4(extrinsic_transform)

    return in_fov(points3d, extrinsic_matrix, cam_intrinsic_matrix, image_size, margin_in_pixels)





