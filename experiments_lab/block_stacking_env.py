from matplotlib import pyplot as plt

from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
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
    def __init__(self,
                 n_blocks,
                 max_steps,
                 ur5e_1_controller: ManipulationController2FG,
                 ur5e_2_controller: ManipulationController2FG,
                 gt: GeometryAndTransforms,
                 camera=None,
                 position_estimator=None,
                 ws_x_lims=workspace_x_lims_default,
                 ws_y_lims=workspace_y_lims_default,
                 stacking_reward=1,
                 finish_ahead_of_time_reward_coeff=0.1,
                 sensing_cost_coeff=0.05,
                 stacking_cost_coeff=0.05):

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
        self.n_picked_blocks = 0

    def reset(self,):
        self.steps = 0
        self.current_tower_height_blocks = 0
        self.n_picked_blocks = 0

        # for safety, just make sure r1 is clear enough
        r1_position = self.r1_controller.getActualTCPPose()[:3]
        assert r1_position[0] > -0.5 and r1_position[1] > -0.5, "r1 is not in a safe position"

        # move r2 to start position
        self.r2_controller.plan_and_move_to_xyzrz(workspace_x_lims_default[1], workspace_y_lims_default[0], 0.15, 0)
        self.current_robot_position = (workspace_x_lims_default[1], workspace_y_lims_default[0])

    def step(self, action: ActionBase) -> tuple[ObservationBase, float]:
        if self.steps >= self.max_steps:
            print("reached max steps, episode is already done")
            return ObservationReachedTerminal(), 0
        if self.n_picked_blocks == self.n_blocks:
            print("all blocks are already picked, episode is done")
            return ObservationReachedTerminal(), 0

        self.steps += 1
        steps_left = self.max_steps - self.steps

        reward = 0

        if isinstance(action, ActionSense):
            height = self.r2_controller.sense_height_tilted(action.x, action.y, start_height=0.1)
            is_occupied = height > 0.035

            observation = ObservationSenseResult(is_occupied,
                                                 robot_position=(action.x, action.y),
                                                 steps_left=steps_left)
            reward -= self.sensing_cost_coeff * \
                        np.linalg.norm(np.asarray(self.current_robot_position) - np.array((action.x, action.y)))

            self.current_robot_position = (action.x, action.y)

        elif isinstance(action, ActionAttemptStack):
            # Note: The observation is whether the block is picked, but the reward is based on the tower height.
            #  It can happen that the block is picked but the tower height remains the same because the block was
            #  picked not in the center, and it falls when the robot tries to put it on the tower.
            #  in that case there is no reward but the agent can update it's belief that the block is not there
            #  anymore.
            self.r2_controller.pick_up(action.x, action.y, 0, start_height=0.15)
            is_picked = self.r2_controller.is_object_gripped()

            if not is_picked:
                success_stack = False
                is_finished = False
            else:
                self.r2_controller.put_down(goal_tower_position[0],
                                            goal_tower_position[1],
                                            0,
                                            start_height=self.n_blocks * 0.04 + 0.1)
                new_tower_height_blocks = self.sense_tower_height()
                success_stack = self.current_tower_height_blocks < new_tower_height_blocks
                self.current_tower_height_blocks = new_tower_height_blocks
                self.n_picked_blocks += 1
                is_finished = self.n_picked_blocks == self.n_blocks

            observed_steps_left = steps_left if not is_finished else 0
            observation = ObservationStackAttemptResult(is_picked,
                                                        robot_position=(action.x, action.y),
                                                        steps_left=observed_steps_left)

            reward += self.stacking_reward * success_stack
            reward += self.finish_ahead_of_time_reward_coeff * is_finished * steps_left
            reward -= self.stacking_cost_coeff * \
                        np.linalg.norm(np.asarray(self.current_robot_position) - np.array((action.x, action.y)))
            reward -= self.stacking_cost_coeff * \
                        np.linalg.norm(np.array(action.x, action.y) - np.asarray(goal_tower_position))

            self.current_robot_position = goal_tower_position

        else:
            raise ValueError(f"unknown action type: {type(action)}")

        return observation, reward

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

