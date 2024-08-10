import logging
from copy import deepcopy

import numpy as np
from frozendict import frozendict

from experiments_lab.experiment_results_data import ExperimentResults
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.manipulation.utils import to_canonical_config, distribute_blocks_in_positions, \
    ur5e_2_distribute_blocks_from_block_positions_dists
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default, goal_tower_position
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_with_depth_as_image
from modeling.belief.block_position_belief import BlockPosDist, BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from experiments_lab.block_stacking_env import LabBlockStackingEnv
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal, ObservationSenseResult, \
    ObservationStackAttemptResult

default_rewards = frozendict(stacking_reward=1,
                             finish_ahead_of_time_reward_coeff=0.1,
                             sensing_cost_coeff=0.1,
                             stacking_cost_coeff=0.2)


class ExperimentManager:
    cleared_blocks_position = [-0.25, -1.15]

    def __init__(self, env: LabBlockStackingEnv, policy: AbastractPolicy):
        self.env = env
        self.policy = policy

        self.ws_x_lims = env.ws_x_lims
        self.ws_y_lims = env.ws_y_lims

        # TODO block piles managements
        # TODO clean up from here, env just manages one experiment

    @classmethod
    def from_params(cls,
                    n_blocks: int,
                    max_steps: int,
                    r1_controller: ManipulationController,
                    r2_controller: ManipulationController,
                    gt: GeometryAndTransforms,
                    camera=None,
                    position_estimator=None,
                    ws_x_lims=workspace_x_lims_default,
                    ws_y_lims=workspace_y_lims_default,
                    rewards: dict = default_rewards, ):
        raise NotImplementedError

    def run_single_experiment(self, init_block_positions,
                              init_block_belief: BlocksPositionsBelief) -> ExperimentResults:
        """
        perform a single experiment with given inital block positions and belief.
        This method places the blocks in the positions from the pile, but it assumes the workspace
        is clear from any other blocks when called.

        @param init_block_positions:
        @param init_block_belief:
        @return: ExperimentResults object with the experiment trajectory and data
        """
        logging.info("running an experiment")
        self.clear_robot1()
        self.distribute_blocks_from_positions(init_block_positions)

        results = ExperimentResults(policy_type=self.policy.__class__.__name__,
                                    agent_params=self.policy.get_params(), )
        results.actual_initial_block_positions = init_block_positions
        results.beliefs.append(init_block_belief)

        self.env.reset()
        self.policy.reset(init_block_belief)

        current_belief = deepcopy(init_block_belief)
        history = []
        accumulated_reward = 0
        for i in range(self.env.max_steps):
            action = self.policy.sample_action(current_belief, history)
            observation, reward = self.env.step(action)

            accumulated_reward += reward

            history.append((action, observation))
            current_belief = self.update_belief(current_belief, action, observation)

            results.beliefs.append(current_belief)
            results.actions.append(action)
            results.observations.append(observation)
            results.rewards.append(reward)

            if isinstance(observation, ObservationReachedTerminal) or len(current_belief.block_beliefs) == 0\
                    or observation.steps_left <= 0:
                break

        results.total_reward = accumulated_reward

        return results

    @staticmethod
    def update_belief(belief, action, observation):
        """ doesn't change inplace! """
        updated_belief = deepcopy(belief)
        if isinstance(observation, ObservationSenseResult):
            updated_belief.update_from_point_sensing_observation(action.x, action.y, observation.is_occupied)
        elif isinstance(observation, ObservationStackAttemptResult):
            updated_belief.update_from_pickup_attempt(action.x, action.y, observation.is_object_picked)
        return updated_belief


    def clear_robot1(self):
        """ make sure robot1 is in a configuration it can never collide with robot2 while it works"""
        self.env.r1_controller.plan_and_move_home()

    def clear_robot2(self):
        """ make sure robot2 is in a configuration it can never collide with robot1 while it works"""
        self.env.r2_controller.plan_and_move_to_xyzrz(workspace_x_lims_default[1], workspace_y_lims_default[0],
                                                      0.15, 0)

    def distribute_blocks_from_positions(self, block_positions):
        """ distribute blocks from given positions """
        logging.info("distributing blocks from positions")

        self.clear_robot1()
        distribute_blocks_in_positions(block_positions, self.env.r2_controller)
        self.clear_robot2()

    def distribute_blocks_from_priors(self, block_positions_distributions: list[BlockPosDist]) -> np.ndarray:
        """
        sample block positions from distribution, and place actual blocks from the pike at those positions.
        returns the sampled block positions
        """
        logging.info("distributing blocks from priors")

        self.clear_robot1()
        block_positions = ur5e_2_distribute_blocks_from_block_positions_dists(block_positions_distributions,
                                                                              self.env.r2_controller)
        self.clear_robot2()

        return block_positions

    def clean_up_workspace(self) -> np.ndarray:
        """
        returns plot of the detections and images of the workspace for block collection
        right now it's simple, throw the blocks in the workspace after taking image from one fixed sensor config
        # TODO, this is what Adi is working on, it should be another module: ExperimentMgr
        """
        logging.info("cleaning up workspace")

        # first, clean the tower:
        tower_height_blocks = self.env.sense_tower_height()
        logging.info(f"tower height is {tower_height_blocks}")
        for i in range(tower_height_blocks):
            start_height = 0.04 * (tower_height_blocks - i) + 0.12
            self.env.r2_controller.pick_up(goal_tower_position[0],
                                           goal_tower_position[1],
                                           0,
                                           start_height)
            self.env.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                          self.cleared_blocks_position[1],
                                                          z=0,
                                                          rz=0)
            self.env.r2_controller.release_grasp()

        self.clear_robot2()

        # now clean other blocks that remain on the table, first detect them:
        lookat = [np.mean(self.ws_x_lims), np.mean(self.ws_y_lims), 0]
        lookat[0] += 0.15
        lookat[1] += 0.15
        clean_up_sensor_config = lookat_verangle_horangle_distance_to_robot_config(lookat,
                                                                                   vertical_angle=60,
                                                                                   horizontal_angle=30,
                                                                                   distance=0.7,
                                                                                   gt=self.env.gt,
                                                                                   robot_name="ur5e_1")
        clean_up_sensor_config = to_canonical_config(clean_up_sensor_config)

        self.env.r1_controller.plan_and_moveJ(clean_up_sensor_config)
        im, depth = self.env.camera.get_frame_rgb()
        positions, annotations = self.env.position_estimator.get_block_positions_depth(im, depth,
                                                                                       clean_up_sensor_config)
        plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                       workspace_x_lims_default, workspace_y_lims_default)
        self.clear_robot1()
        logging.info(f"detected {len(positions)} blocks")

        # now clean the blocks:
        for p in positions:
            self.env.r2_controller.pick_up(p[0], p[1], 0, start_height=0.15)
            self.env.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                          self.cleared_blocks_position[1],
                                                          z=0,
                                                          rz=0)
            self.env.r2_controller.release_grasp()

        logging.info("workspace cleaning procedure finished")

        return plot_im
