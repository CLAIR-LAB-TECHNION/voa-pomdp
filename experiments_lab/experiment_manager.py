import logging
import os
from copy import deepcopy
import time

import cv2
import numpy as np
from frozendict import frozendict
from experiments_lab.experiment_results_data import ExperimentResults
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.manipulation.utils import to_canonical_config, distribute_blocks_in_positions, \
    ur5e_2_distribute_blocks_from_block_positions_dists, ur5e_2_collect_blocks_from_positions
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default, goal_tower_position, \
    sample_block_positions_from_dists
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_with_depth_as_image
from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from modeling.belief.block_position_belief import BlockPosDist, BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
from experiments_lab.block_stacking_env import LabBlockStackingEnv
from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal, ObservationSenseResult, \
    ObservationStackAttemptResult
from modeling.sensor_distribution import detections_to_distributions

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

        self.help_configs = None
        # TODO block piles managements

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
        # TODO for convenience
        raise NotImplementedError

    def run_single_experiment(self,
                              init_block_positions,
                              init_block_belief: BlocksPositionsBelief,
                              help_config=None,
                              help_detections_filename=None,
                              plot_beliefs=False) -> ExperimentResults:
        """
        perform a single experiment with given inital block positions and belief.
        This method places the blocks in the positions from the pile, but it assumes the workspace
        is clear from any other blocks when called.

        @param init_block_positions:
        @param init_block_belief:
        @param help_config: config of robot1 to take image from to help. if None, there is no help in that experiment
        @param help_detections_filename: if help_config is not None, this is the filename to save the detections to
        @param plot_beliefs: if True, plot the belief at each step
        @return: ExperimentResults object with the experiment trajectory and data
        """
        logging.info("running an experiment")
        self.clear_robot1()
        self.safe_distribute_blocks_in_positions(init_block_positions)

        results = ExperimentResults(policy_type=self.policy.__class__.__name__,
                                    agent_params=self.policy.get_params(),
                                    help_config=help_config)
        results.actual_initial_block_positions = init_block_positions
        if help_config is not None:
            results.is_with_help = True
            results.help_config = help_config
            results.belief_before_help = deepcopy(init_block_belief)
            detection_muss, detections_sigmas, detections_im = \
                self.help_and_update_belief(init_block_belief, help_config)
            if help_detections_filename is not None:
                cv2.imwrite(help_detections_filename, cv2.cvtColor(detections_im, cv2.COLOR_RGB2BGR))

            results.help_detections_mus = detection_muss
            results.help_detections_sigmas = detections_sigmas

        results.beliefs.append(init_block_belief)

        self.env.reset()
        self.policy.reset(init_block_belief)

        current_belief = deepcopy(init_block_belief)
        history = []
        accumulated_reward = 0
        for i in range(self.env.max_steps):
            if plot_beliefs:
                self.plot_belief(current_belief, history)
            action = self.policy.sample_action(current_belief, history)
            observation, reward = self.env.step(action)

            accumulated_reward += reward

            history.append((action, observation))
            current_belief = self.update_belief(current_belief, action, observation)

            results.beliefs.append(current_belief)
            results.actions.append(action)
            results.observations.append(observation)
            results.rewards.append(reward)

            if isinstance(observation, ObservationReachedTerminal) or len(current_belief.block_beliefs) == 0 \
                    or observation.steps_left <= 0:
                break

        results.total_reward = accumulated_reward

        return results

    def help_and_update_belief(self, block_belief: BlocksPositionsBelief, help_config) \
            -> (np.ndarray, np.ndarray, np.ndarray):
        """
        takes image with robot1 from help config, detects blocks and uses the detections to update
        the belief **inplace**. This method also returns the expectations and stds of the detections,
        along with an image of the detections
        """
        # make sure robot2 is clear:
        self.clear_robot2()

        self.env.r1_controller.plan_and_moveJ(help_config)
        im, depth = self.env.camera.get_frame_rgb()
        positions, annotations = self.env.position_estimator.get_block_positions_depth(im, depth, help_config)
        detections_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                             workspace_x_lims_default, workspace_y_lims_default)

        camera_position = self.env.r1_controller.getActualTCPPose()[:3]
        mus, sigmas = detections_to_distributions(positions, camera_position)

        if len(mus) == 0:
            return [], [], detections_im

        ordered_detection_mus, ordered_detection_sigmas = \
            block_belief.update_from_image_detections_position_distribution(mus, sigmas)

        return ordered_detection_mus, ordered_detection_sigmas, detections_im

    @staticmethod
    def update_belief(belief, action, observation):
        """ doesn't change inplace! """
        updated_belief = deepcopy(belief)
        if isinstance(observation, ObservationSenseResult):
            updated_belief.update_from_point_sensing_observation(action.x, action.y, observation.is_occupied)
        elif isinstance(observation, ObservationStackAttemptResult):
            updated_belief.update_from_pickup_attempt(action.x, action.y, observation.is_object_picked)
        return updated_belief

    def run_value_difference_experiments(self,
                                         init_block_positions,
                                         init_block_belief: BlocksPositionsBelief,
                                         helper_config,
                                         dirname):
        """
        run two experiments, one with help and one without, and save results to directory with dirname,
        in an internal directory with datetime stamp
        both experiments start with the same start state, but the belief in the experiment with help
        is updated after taking image from helper_config before the experiment starts.

        It is assumed the workspace is clear before the experiment starts and that there are enough blocks
        in the pile
        @param init_block_positions:
        @param init_block_belief:
        @param helper_config:
        @param dirname:
        @return:
        """
        datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"{dirname}/{datetime_stamp}")

        logging.info(f"starting value diff experiment {datetime_stamp}")

        resutls_no_help = self.run_single_experiment(init_block_positions, init_block_belief, None)
        resutls_no_help.save(f"{dirname}/{datetime_stamp}/no_help.pkl")

        no_help_cleanup_detections = self.clean_up_workspace()
        cv2.imwrite(f"{dirname}/{datetime_stamp}/no_help_cleanup.png", no_help_cleanup_detections)

        results_with_help = self.run_single_experiment(init_block_positions, init_block_belief, helper_config,
                                                       f"{dirname}/{datetime_stamp}/help_detections.png")
        results_with_help.save(f"{dirname}/{datetime_stamp}/with_help.pkl")

        with_help_cleanup_detections = self.clean_up_workspace()
        cv2.imwrite(f"{dirname}/{datetime_stamp}/with_help_cleanup.png", with_help_cleanup_detections)

        logging.info(f"value difference experiment is over, all data saved {datetime_stamp}")

    def sample_value_difference_experiments(self,
                                            n_blocks,
                                            min_prior_std,
                                            max_prior_std,
                                            dirname,
                                            ):
        """
        Run two experiment from the same initial belief and start state, once with help, and once without.
        The initial belief is sampled from a truncated gaussian belief space, where the location is sample
        uniformly in the workspace for each block, and the scale is sampled from [max_prior_std, min_prior_std]
        and initial state is sampled from that belief.
        This method assumes workspace is clear at the beginning, and tries to clean up after the experiments
        are done.
        @param n_blocks: number of blocks in the experiment, which determines how many blocks will be in the
            initial belief and initial state
        @param min_prior_std: minimal std for the prior belief for each block
        @param max_prior_std: maximal std for the prior belief for each block
        @param dirname: directory to save the results to, everything will be saved in a
            subdirectory with datetime stamp
        @return: Nothing
        """

        # sample blocks mus and sigmas:
        # mus for x y between workspace limits
        block_pos_mu = np.random.uniform(low=[self.ws_x_lims[0], self.ws_y_lims[0]],
                                         high=[self.ws_x_lims[1], self.ws_y_lims[1]],
                                         size=(n_blocks, 2))
        block_pos_sigma = np.random.uniform(low=min_prior_std, high=max_prior_std, size=(n_blocks, 2))

        init_block_belief = BlocksPositionsBelief(n_blocks, self.ws_x_lims, self.ws_y_lims,
                                                  init_mus=block_pos_mu, init_sigmas=block_pos_sigma)
        init_block_positions = sample_block_positions_from_dists(init_block_belief.block_beliefs)

        help_config = self.sample_help_config()

        self.run_value_difference_experiments(init_block_positions=init_block_positions,
                                              init_block_belief=init_block_belief,
                                              helper_config=help_config,
                                              dirname=dirname)

    def sample_help_config(self):
        """
        sample config for robot1 to take image from to help.
        the config is sampled from a file of configs that were sampled according to a heuristic
        where the camera looks at some point at the workspace from different angles and distances
        @return: Configuration for robot1, with canonical joint angels (between -pi and pi)
        """
        if self.help_configs is None:
            self.help_configs = np.load("help_configs.npy")
            logging.debug(f"{len(self.help_configs)} help configs loaded")

        help_config = self.help_configs[np.random.randint(len(self.help_configs))]

        return help_config

    def clear_robot1(self):
        """ make sure robot1 is in a configuration it can never collide with robot2 while it works"""
        self.env.r1_controller.plan_and_move_home()

    def clear_robot2(self):
        """ make sure robot2 is in a configuration it can never collide with robot1 while it works"""
        self.env.r2_controller.plan_and_move_to_xyzrz(goal_tower_position[0], goal_tower_position[1],
                                                      (self.env.n_blocks + 2) * 0.04, 0)

    def safe_distribute_blocks_in_positions(self, block_positions):
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

    def clean_up_workspace(self, put_back_to_stack=False) -> np.ndarray:
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
        if put_back_to_stack:
            ur5e_2_collect_blocks_from_positions(positions, self.env.r2_controller)
        else:
            for p in positions:
                self.env.r2_controller.pick_up(p[0], p[1], 0, start_height=0.15)
                self.env.r2_controller.plan_and_move_to_xyzrz(self.cleared_blocks_position[0],
                                                              self.cleared_blocks_position[1],
                                                              z=0,
                                                              rz=0)
                self.env.r2_controller.release_grasp()

        logging.info("workspace cleaning procedure finished")

        return plot_im

    def plot_belief(self, current_belief, history=[]):
        # use the history for points
        positive_sens = []
        negative_sens = []
        failed_pickups = []
        for a, o in history:
            if isinstance(o, ObservationSenseResult):
                if o.is_occupied:
                    positive_sens.append((a.x, a.y))
                else:
                    negative_sens.append((a.x, a.y))
            elif isinstance(o, ObservationStackAttemptResult):
                if not o.is_object_picked:
                    failed_pickups.append((a.x, a.y))

        plot_all_blocks_beliefs(current_belief,
                                positive_sensing_points=positive_sens,
                                negative_sensing_points=negative_sens,
                                pickup_attempt_points=failed_pickups)
