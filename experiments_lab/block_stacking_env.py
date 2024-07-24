from matplotlib import pyplot as plt

from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.manipulation.utils import ur5e_2_distribute_blocks_from_block_positions_dists,\
    distribute_blocks_in_positions
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default)
import numpy as np
import logging

from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_with_depth_as_image


class LabBlockStackingEnv:
    cleared_blocks_position = [-0.25, -1.15]
    goal_tower_position = [-0.45, -1.15]

    def __init__(self,
                 n_blocks,
                 max_steps,
                 ur5e_1_controller: ManipulationController,
                 ur5e_2_controller: ManipulationController,
                 gt: GeometryAndTransforms,
                 camera=None,
                 position_estimator=None,
                 ws_x_lims=workspace_x_lims_default,
                 ws_y_lims=workspace_y_lims_default):

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

        self.steps = 0
        self.accumulated_cost = 0

        # will update at first reset:
        self._block_positions = None  # should be hidden from the agent!

    def reset_from_distribution(self, block_positions_distributions):
        logging.info("reseting env")
        self._reset()
        self._block_positions = ur5e_2_distribute_blocks_from_block_positions_dists(block_positions_distributions,
                                                                                    self.r2_controller)
        self.r2_controller.plan_and_move_to_xyzrz(workspace_x_lims_default[0], workspace_y_lims_default[1], 0.15, 0)
        logging.info(f"env rested, blocks are put at positions: {self._block_positions}")

    def reset_from_positions(self, block_positions):
        logging.info("reseting env")
        self._reset()
        self._block_positions = block_positions
        distribute_blocks_in_positions(block_positions, self.r2_controller)
        self.r2_controller.plan_and_move_to_xyzrz(workspace_x_lims_default[0], workspace_y_lims_default[1], 0.15, 0)
        logging.info(f"env rested, blocks are put at positions: {self._block_positions}")

    def _reset(self):
        self.steps = 0
        self.accumulated_cost = 0

        self.clean_workspace_for_next_experiment()

        # clear r1 if it's not home
        self.r1_controller.plan_and_move_home()

    def step(self, action_type, x, y):
        assert action_type in ["sense", "attempt_stack"]
        if self.steps >= self.max_steps:
            print("reach max steps, episode is already done")
            return None

        self.steps += 1
        steps_left = self.max_steps - self.steps
        self.accumulated_cost += self._get_action_cost(action_type, x, y)

        if action_type == "sense":
            height = self.r1_controller.sense_height_tilted(x, y)
            is_occupied = height > 0.03
            observation = (is_occupied, steps_left)
        else:
            self.r2_controller.pick_up(x, y, 0.15)
            # success = self.r2_controller.measure_weight() TODO
            pick_success = True  # assume now success
            self.r2_controller.put_down(x, y, start_height=self.n_blocks*0.04 + 0.1)
            observation = (pick_success, steps_left)

        return observation

    def clean_workspace_for_next_experiment(self):
        """
        right now it's simple, throw the blocks in the workspace after picturing from one fixed sensor config
        # TODO, this is what Adi is working on, it should be another module: ExperimentMgr
        """
        # first, clean the tower:
        tower_height_blocks = self.sense_tower_height()
        for i in range(tower_height_blocks):
            start_height = 0.04 * (tower_height_blocks - i) + 0.1
            self.r2_controller.pick_up(self.goal_tower_position[0], self.goal_tower_position[1], start_height)
            self.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                      self.cleared_blocks_position[1],
                                                      z=0,
                                                      rz=0)
            self.r2_controller.release_grasp()

        # now clean other blocks that remain on the table, first detect them:
        lookat = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0]
        clean_up_sensor_config = lookat_verangle_horangle_distance_to_robot_config(lookat,
                                                                                   horizontal_angle=60,
                                                                                   vertical_angle=30,
                                                                                   distance=0.65,
                                                                                   gt=self.r1_controller.gt,
                                                                                   robot_name="ur5e_1")
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
            self.r2_controller.pick_up(p[0], p[1], 0.15)
            self.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                      self.cleared_blocks_position[1],
                                                      z=0,
                                                      rz=0)
            self.r2_controller.release_grasp()

    def sense_tower_height(self):
        max_height = 0.04 * self.n_blocks
        start_heigh = max_height + 0.1

        height = self.r1_controller.sense_height_tilted(self.goal_tower_position[0],
                                                        self.goal_tower_position[1],
                                                        start_heigh)

        # first block should be at about 0.032, then 0.04 for every other block
        if height < 0.03:
            return 0

        n_blocks_float = ((height - 0.032) / 0.04) + 1
        # go for the ceil, worst case we will do one redundant pick
        n_blocks = int(np.ceil(n_blocks_float))
        return n_blocks

    def _get_action_cost(self, action_type, x, y):
        return 0  # TODO, need to model that in the pomdp as well, maybe create cost model...













