import numpy as np
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from modeling.voa_predictions.utils import in_r1_fov
from modeling.sensor_distribution import detections_to_distributions, compute_position_stds


def sample_observation_fov_based(block_pos_state,
                                 help_config,
                                 gt: GeometryAndTransforms,
                                 cam_intrinsic_matrix,
                                 detection_probability=0.9,
                                 margin_in_pixels=30,):
    """
    sample observation given the block positions and the helper configuration.
    assuming blocks that are within fov of the camera are detected with a probability of detection_probability,
    ignoring occulusions.
    the detected mus for the positins have added noise base on the same model as in detections_to_distributions

    @param block_pos_state: state (list of block positions)
    @param help_config: config of robot with the camera
    @param gt: GeometryAndTransforms object for that scene
    @param cam_intrinsic_matrix: camera intrinsic matrix
    @param detection_probability: probability of detecting blocks that are in fov
    @param margin_in_pixels: blocks that are in fov but close to the edge by this margin in one of the axes,
        are considered out of fov
    @return:
    """
    plane_z = 0.025

    block_pos_3d = np.concatenate([block_pos_state, plane_z * np.ones((len(block_pos_state), 1))], axis=1)
    blocks_in_fov = in_r1_fov(block_pos_3d, help_config, gt, cam_intrinsic_matrix, margin_in_pixels=margin_in_pixels)

    # block that are in fov are detected with a probability of detection_probability
    blocks_detected_if_in_fov = np.random.rand(len(block_pos_state)) < detection_probability
    blocks_detected = blocks_in_fov & blocks_detected_if_in_fov

    block_pos_state = np.asarray(block_pos_state)
    detected_blocks_positions = block_pos_state[blocks_detected]

    camera_position = gt.point_camera_to_world(point_camera=np.array([0, 0, 0]),
                                               robot_name="ur5e_1",
                                               config=help_config)

    # add std computation to external func? and add that noise to detection
    detected_blocks_positions_3d = np.concatenate([detected_blocks_positions,
                                                   plane_z * np.ones((len(detected_blocks_positions), 1))], axis=1)
    # noise_stds = compute_position_stds(detected_blocks_positions_3d, camera_position)
    # detected_blocks_positions += np.random.randn(*detected_blocks_positions.shape) * noise_stds

    return detections_to_distributions(detected_blocks_positions_3d, camera_position)
