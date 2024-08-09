from matplotlib import pyplot as plt

from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.manipulation.utils import ur5e_2_distribute_blocks_from_block_positions_dists, \
    distribute_blocks_in_positions, to_canonical_config
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)
import numpy as np
import logging
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_with_depth_as_image
from modeling.pomdp_problem.domain.observation import *
from modeling.pomdp_problem.domain.action import *


class LabBlockStackingEnv:
    cleared_blocks_position = [-0.25, -1.15]

    def __init__(self,
                 n_blocks,
                 max_steps,
                 ur5e_1_controller: ManipulationController,
                 ur5e_2_controller: ManipulationController,
                 gt: GeometryAndTransforms,
                 camera=None,
                 position_estimator=None,
                 ws_x_lims=workspace_x_lims_default,
                 ws_y_lims=workspace_y_lims_default,
                 stacking_reward=1,
                 finish_ahead_of_time_reward_coeff=0.1,
                 sensing_cost_coeff=0.1,
                 stacking_cost_coeff=0.2):

        self.n_blocks = n_blocks
        self.max_steps = max_steps
        self.r1_controller = ur5e_1_controller
        self.r2_controller = ur5e_2_controller
        self.gt = gt
        self.ws_x_lims = ws_x_lims
        self.ws_y_lims = ws_y_lims
        self.camera = camera if camera is not None else RealsenseCamera()
        self.position_estimator = position_estimator if position_estimator is not None \
            else ImageBlockPositionEstimator(ws_x_lims, ws_y_lims, gt)
        self.stacking_reward = stacking_reward
        self.finish_ahead_of_time_reward_coeff = finish_ahead_of_time_reward_coeff
        self.sensing_cost_coeff = sensing_cost_coeff
        self.stacking_cost_coeff = stacking_cost_coeff

        self.steps = 0
        self.current_robot_position = None
        self.current_tower_height_blocks = 0

        # will update at first reset:
        self._block_positions = None  # should be hidden from the agent!

    def reset_from_distribution(self, block_positions_distributions, perform_cleanup=True):
        logging.info("reseting env")
        self._reset(perform_cleanup)
        self._block_positions = ur5e_2_distribute_blocks_from_block_positions_dists(block_positions_distributions,
                                                                                    self.r2_controller)
        self.r1_controller.plan_and_move_home()
        self.r2_controller.plan_and_move_to_xyzrz(workspace_x_lims_default[1], workspace_y_lims_default[0], 0.15, 0)
        self.current_robot_position = (workspace_x_lims_default[1], workspace_y_lims_default[0])
        logging.info(f"env rested, blocks are put at positions: {self._block_positions}")

    def reset_from_positions(self, block_positions, perform_cleanup=True):
        logging.info("reseting env")
        self._reset(perform_cleanup)
        self._block_positions = block_positions
        distribute_blocks_in_positions(block_positions, self.r2_controller)
        self.r1_controller.plan_and_move_home()
        self.r2_controller.plan_and_move_to_xyzrz(workspace_x_lims_default[1], workspace_y_lims_default[0], 0.15, 0)
        self.current_robot_position = (workspace_x_lims_default[1], workspace_y_lims_default[0])
        logging.info(f"env rested, blocks are put at positions: {self._block_positions}")

    def _reset(self, perform_cleanup=True):
        self.steps = 0

        if perform_cleanup:
            self.clean_workspace_for_next_experiment()

        # clear r1 if it's not home
        self.r1_controller.plan_and_move_home()
        self.current_tower_height_blocks = 0

    def step(self, action: ActionBase) -> tuple[ObservationBase, float]:
        if self.steps >= self.max_steps:
            print("reached max steps, episode is already done")
            return ObservationReachedTerminal(), 0

        self.steps += 1
        steps_left = self.max_steps - self.steps

        reward = 0

        if isinstance(action, ActionSense):
            height = self.r2_controller.sense_height_tilted(action.x, action.y, start_height=0.12)
            is_occupied = height > 0.036

            observation = ObservationSenseResult(is_occupied,
                                                 robot_position=(action.x, action.y),
                                                 steps_left=steps_left)
            reward -= self.sensing_cost_coeff * \
                        np.linalg.norm(np.asarray(self.current_robot_position) - np.array((action.x, action.y)))

            self.current_robot_position = (action.x, action.y)

        elif isinstance(action, ActionAttemptStack):
            self.r2_controller.pick_up(action.x, action.y, 0.12)
            self.r2_controller.put_down(goal_tower_position[0],
                                        goal_tower_position[1],
                                        0,
                                        start_height=self.n_blocks * 0.04 + 0.1)
            new_tower_height_blocks = self.sense_tower_height()
            success_stack = self.current_tower_height_blocks < new_tower_height_blocks
            self.current_tower_height_blocks = new_tower_height_blocks
            is_finished = self.current_tower_height_blocks == self.n_blocks

            observed_steps_left = steps_left if not is_finished else 0
            observation = ObservationStackAttemptResult(success_stack,
                                                        robot_position=(action.x, action.y),
                                                        steps_left=observed_steps_left)

            reward += self.stacking_reward * success_stack
            reward += self.finish_ahead_of_time_reward_coeff * is_finished * steps_left
            reward -= self.stacking_cost_coeff * \
                        np.linalg.norm(np.asarray(self.current_robot_position) - np.array((action.x, action.y)))
            reward -= self.stacking_cost_coeff * \
                        np.linalg.norm(np.arrary(action.x, action.y) - np.asarray(goal_tower_position))

            self.current_robot_position = goal_tower_position

        else:
            raise ValueError(f"unknown action type: {type(action)}")

        return observation, reward

    def clean_workspace_for_next_experiment(self):
        """
        right now it's simple, throw the blocks in the workspace after picturing from one fixed sensor config
        # TODO, this is what Adi is working on, it should be another module: ExperimentMgr
        """
        # first, clean the tower:
        tower_height_blocks = self.sense_tower_height()
        for i in range(tower_height_blocks):
            start_height = 0.04 * (tower_height_blocks - i) + 0.12
            self.r2_controller.pick_up(goal_tower_position[0],
                                       goal_tower_position[1],
                                       0,
                                       start_height)
            self.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                      self.cleared_blocks_position[1],
                                                      z=0,
                                                      rz=0)
            self.r2_controller.release_grasp()

        # now clean other blocks that remain on the table, first detect them:
        lookat = [np.mean(self.ws_x_lims), np.mean(self.ws_y_lims), 0]
        lookat[0] += 0.15
        lookat[1] += 0.15
        clean_up_sensor_config = lookat_verangle_horangle_distance_to_robot_config(lookat,
                                                                                   vertical_angle=60,
                                                                                   horizontal_angle=30,
                                                                                   distance=0.7,
                                                                                   gt=self.r1_controller.gt,
                                                                                   robot_name="ur5e_1")
        clean_up_sensor_config = to_canonical_config(clean_up_sensor_config)

        self.r1_controller.plan_and_moveJ(clean_up_sensor_config)
        im, depth = self.camera.get_frame_rgb()
        positions, annotations = self.position_estimator.get_block_positions_depth(im, depth, clean_up_sensor_config)
        plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                       workspace_x_lims_default, workspace_y_lims_default,
                                                       actual_positions=self._block_positions)
        plt.figure(figsize=(12, 12), dpi=512)
        plt.imshow(plot_im)
        plt.axis('off')
        plt.title("cleanup")
        plt.tight_layout()
        plt.show()
        self.r1_controller.plan_and_move_home()

        # now clean the blocks:
        for p in positions:
            self.r2_controller.pick_up(p[0], p[1], 0, start_height=0.15)
            self.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                      self.cleared_blocks_position[1],
                                                      z=0,
                                                      rz=0)
            self.r2_controller.release_grasp()

    def sense_tower_height(self):
        max_height = 0.04 * self.n_blocks
        start_heigh = max_height + 0.12

        height = self.r2_controller.sense_height_tilted(goal_tower_position[0],
                                                        goal_tower_position[1],
                                                        start_heigh)

        n_blocks = 0
        # first block should be at about 0.0395, then 0.04 for every other block
        while height > 0.03:  # for sure there are blocks there, and for sure not when condition is not met
            n_blocks += 1
            height -= 0.04

        return n_blocks

