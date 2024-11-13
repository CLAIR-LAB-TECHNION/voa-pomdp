import cv2
import numpy as np
from experiments_sim.block_stacking_simulator import BlockStackingSimulator, helper_camera_translation_from_ee
from experiments_sim.shared_position_estimator import SharedImageBlockPositionEstimator
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.vision.utils import detections_plots_no_depth_as_image
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.sensor_distribution import detections_to_distributions


def build_position_estimator(env: BlockStackingSimulator, shared=False) \
        -> (ImageBlockPositionEstimator, GeometryAndTransforms):
    gt = GeometryAndTransforms(env.motion_executor.motion_planner,
                               cam_in_ee=-np.array(env.helper_camera_translation_from_ee))
    if shared:
        position_estimator = SharedImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt,
                                                               "ur5e_1",
                                                               intrinsic_camera_matrix=env.mujoco_env.get_robot_cam_intrinsic_matrix())
    else:
        position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt,
                                                         "ur5e_1",
                                                         intrinsic_camera_matrix=env.mujoco_env.get_robot_cam_intrinsic_matrix())
    return position_estimator, gt


def help_and_update_belief(env: BlockStackingSimulator, belief: BlocksPositionsBelief, help_config,
                           position_estimator: ImageBlockPositionEstimator, hidden_actual_positions=None) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
    moves robot, takes image, updates belief INPLACE
    @param env:
    @param belief:
    @param help_config:
    @param position_estimator: ImageBlockPositionEstimator object that is used for belief update
    @param hidden_actual_positions: actual positions of the blocks to show on the plot. Of course it's not used
        for belief update, but for visualization purposes.
    @return: detections mus used to update the belief, detection sigmas, and the image of detections
    """
    env.clear_r2_no_motion()
    im_rgb, actual_config = env.sense_camera_r1(help_config)
    im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    pred_positions, annotations = position_estimator.get_block_position_plane_projection(im, actual_config,
                                                                                         plane_z=0.024,
                                                                                         max_detections=4)

    detections_plot = detections_plots_no_depth_as_image(annotations[0], annotations[1], pred_positions,
                                                         workspace_x_lims_default, workspace_y_lims_default,
                                                         actual_positions=hidden_actual_positions)

    # filter positions that are outside the workspace
    pred_positions = [pos for pos in pred_positions if
                      workspace_x_lims_default[0] < pos[0] < workspace_x_lims_default[1]
                      and workspace_y_lims_default[0] < pos[1] < workspace_y_lims_default[1]]

    camera_position = position_estimator.gt.point_camera_to_world(point_camera=np.array([0, 0, 0]),
                                                                  robot_name="ur5e_1",
                                                                  config=actual_config)

    mus, sigmas = detections_to_distributions(pred_positions, camera_position)

    if len(mus) == 0:
        return [], [], detections_plot

    ordered_detection_mus, ordered_detection_sigmas = \
        belief.update_from_image_detections_position_distribution(mus, sigmas)

    return ordered_detection_mus, ordered_detection_sigmas, detections_plot
