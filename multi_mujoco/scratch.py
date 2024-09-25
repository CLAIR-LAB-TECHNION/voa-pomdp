import cv2
import numpy as np
from matplotlib import pyplot as plt
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.vision.utils import detections_plots_no_depth_as_image
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack


from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator


blocks_pos = [[-0.8, -0.8],
              [-0.8, -0.7],
              [-0.65, -0.72],
              [-0.6, -0.6],]

# blocks in 4 corners of the workspace:
# blocks_pos = [[workspace_x_lims_default[0], workspace_y_lims_default[0]],
#                 [workspace_x_lims_default[0], workspace_y_lims_default[1]],
#                 [workspace_x_lims_default[1], workspace_y_lims_default[0]],
#                 [workspace_x_lims_default[1], workspace_y_lims_default[1]]]

simulator = BlockStackingSimulator(visualize_mp=False, max_steps=7, render_sleep_to_maintain_fps=False,
                                   render_mode='human')

gt = GeometryAndTransforms(simulator.motion_executor.motion_planner, cam_in_ee=-np.array(simulator.helper_camera_translation_from_ee))
position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt, "ur5e_1",
                                                 intrinsic_camera_matrix=simulator.mujoco_env.get_robot_cam_intrinsic_matrix())

help_configs = np.load("../experiments_lab/configurations/help_configs.npy")

for i in range(10):
    blocks_pos[0][0] += 0.01
    print("resetting with blocks_pos", blocks_pos)
    simulator.reset(block_positions=blocks_pos)

    im, actual_config_for_im = simulator.sense_camera_r1([-0.5567808869537451, -1.215127556940363, -1.8444974285294637, 1.258726315197987, -0.6591467161558847, -1.20351355710407049])
    # plt.imshow(im)
    # plt.show()

    pred_positions, annotations = position_estimator.get_block_position_plane_projection(im, actual_config_for_im,
                                                                                    plane_z=0.024)
    detections_plot = detections_plots_no_depth_as_image(annotations[0], annotations[1], pred_positions,
                                                         workspace_x_lims_default, workspace_y_lims_default,
                                                         actual_positions=blocks_pos)

    plt.figure(dpi=512, tight_layout=True, figsize=(5, 10))

    plt.imshow(detections_plot)
    plt.axis('off')
    plt.show()

    obs = None

    while True:
        # sample action:
        i = np.random.randint(0, 2)
        x, y = np.random.uniform(-0.9, -0.54), np.random.uniform(-1.0, -0.55)
        action = ActionSense(x, y) if i == 0 else ActionAttemptStack(x, y)
        obs, reward = simulator.step(action)
        print("-----------------")
        print(action)
        print(obs)
        print("reward", reward)

        if obs.steps_left == 0:
            break
