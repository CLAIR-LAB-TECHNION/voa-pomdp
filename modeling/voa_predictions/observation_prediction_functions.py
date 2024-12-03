import numpy as np
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from modeling.voa_predictions.utils import in_r1_fov
from modeling.sensor_distribution import detections_to_distributions, compute_position_stds


def sample_observation_fov_based(block_pos_state,
                                 help_config,
                                 gt: GeometryAndTransforms,
                                 cam_intrinsic_matrix,
                                 detection_probability_at_max_distance=0.5,
                                 max_distance=1.5,
                                 margin_in_pixels=30,
                                 detection_noise_scale=0.1):
    """
    Sample observation given block positions and helper configuration.
    Detection probability varies linearly with distance from the camera:
    - 100% detection probability at minimum distance (0.35m) and closer
    - detection_probability_at_max_distance at maximum distance and beyond
    - Linear falloff between min and max distances
    Occlusions are ignored in this model.
    Detected positions have added noise based on the same model as in detections_to_distributions scaled by detection_noise_scale.

    Args:
        block_pos_state: State represented as list of block positions (x,y coordinates)
        help_config: Configuration of robot with the camera
        gt: GeometryAndTransforms object for the scene
        cam_intrinsic_matrix: Camera intrinsic matrix
        detection_probability_at_max_distance: Detection probability at and beyond max_distance (default: 0.9)
        max_distance: Maximum distance threshold in meters (default: 1.5m)
        margin_in_pixels: Blocks within this margin from image edges are considered out of FOV (default: 30)
        detection_noise_scale: Scale factor for detection noise
    Returns:
        Distributions derived from the detected block positions using detections_to_distributions()
    """

    plane_z = 0.025
    min_distance = 0.35

    camera_position = gt.point_camera_to_world(point_camera=np.array([0, 0, 0]),
                                               robot_name="ur5e_1",
                                               config=help_config)

    block_pos_3d = np.concatenate([block_pos_state, plane_z * np.ones((len(block_pos_state), 1))], axis=1)
    are_blocks_in_fov = in_r1_fov(block_pos_3d, help_config, gt, cam_intrinsic_matrix, margin_in_pixels=margin_in_pixels)
    blocks_pos_3d_in_fov = np.asarray(block_pos_3d[are_blocks_in_fov])

    # distance from camera for blocks in fov:
    dists = np.linalg.norm(blocks_pos_3d_in_fov - camera_position, axis=1)
    # probability of detection is a function of distance, 100% detection at min distance and on,
    # detection_probability_at_max_distance at max distance and on
    distance_ratio = (dists - min_distance) / (max_distance - min_distance)
    clamped_distance_ratio = np.clip(distance_ratio, 0, 1)
    probability_falloff = 1 - clamped_distance_ratio
    detection_probability = (detection_probability_at_max_distance +
                             (1 - detection_probability_at_max_distance) * probability_falloff)

    are_blocks_detected = np.random.rand(len(blocks_pos_3d_in_fov)) < detection_probability

    detected_blocks_positions_3d = blocks_pos_3d_in_fov[are_blocks_detected]

    noise_stds = detection_noise_scale * compute_position_stds(detected_blocks_positions_3d, camera_position)

    # add noise just to x and y:
    detected_blocks_positions_3d[:, :2] += np.random.randn(*detected_blocks_positions_3d[:, :2].shape) * noise_stds

    return detections_to_distributions(detected_blocks_positions_3d, camera_position)
