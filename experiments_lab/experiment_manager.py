import logging
import os
from copy import deepcopy, copy
import time

import cv2
import numpy as np
import pandas as pd
from frozendict import frozendict
from experiments_lab.experiment_results_data import ExperimentResults
from experiments_lab.experiment_visualizer import ExperimentVisualizer
from experiments_lab.utils import plot_belief_with_history
from lab_ur_stack.camera.realsense_camera import RealsenseCameraWithRecording
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.manipulation.utils import to_canonical_config,  ur5e_2_collect_blocks_from_positions
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

    def __init__(self, env: LabBlockStackingEnv, policy: AbastractPolicy, visualize=True):
        self.env = env
        self.policy = policy

        self.ws_x_lims = env.ws_x_lims
        self.ws_y_lims = env.ws_y_lims

        self.help_configs = None
        self.piles_manager = BlockPilesManager()

        self.visualizer = ExperimentVisualizer() if visualize else None

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

        if self.visualizer is not None:
            belief_im = plot_all_blocks_beliefs(init_block_belief,
                                                actual_states=init_block_positions,
                                                ret_as_image=True)
            self.visualizer.update_belief_image(belief_im)

        results = ExperimentResults(policy_type=self.policy.__class__.__name__,
                                    agent_params=self.policy.get_params(),
                                    help_config=help_config)
        results.actual_initial_block_positions = init_block_positions
        if help_config is not None:
            results.is_with_help = True
            results.help_config = help_config
            results.belief_before_help = deepcopy(init_block_belief)
            detection_mus, detections_sigmas, detections_im = \
                self.help_and_update_belief(init_block_belief, help_config)
            if help_detections_filename is not None:
                cv2.imwrite(help_detections_filename, cv2.cvtColor(detections_im, cv2.COLOR_RGB2BGR))

            results.help_detections_mus = detection_mus
            results.help_detections_sigmas = detections_sigmas

            if self.visualizer is not None:
                self.visualizer.update_detection_image(detections_im, "Help Detections")
                self.visualizer.update_belief_image(plot_belief_with_history(init_block_belief,
                                                                            actual_state=init_block_positions,
                                                                            ret_as_image=True))
                observed_mus_and_sigmas = [(detection_mus[i], detections_sigmas[i]) for i in range(len(detection_mus))]
                self.visualizer.add_detections_distributions(plot_belief_with_history(results.belief_before_help,
                                                                                      actual_state=init_block_positions,
                                                                                      observed_mus_and_sigmas=observed_mus_and_sigmas,
                                                                                      ret_as_image=True))
        else:
            results.is_with_help = False
            self.visualizer.add_detections_distributions(plot_all_blocks_beliefs(init_block_belief,
                                                                               actual_states=init_block_positions,
                                                                               ret_as_image=True)) # initial belief


        results.beliefs.append(init_block_belief)

        self.env.reset()
        self.policy.reset(init_block_belief)

        current_belief = deepcopy(init_block_belief)
        history = []
        accumulated_reward = 0
        for i in range(self.env.max_steps):
            if plot_beliefs:
                plot_belief_with_history(current_belief, actual_state=init_block_positions, history=history)
            action = self.policy.sample_action(current_belief, history)
            observation, reward = self.env.step(action)

            accumulated_reward += reward

            history.append((action, observation))
            current_belief = self.update_belief(current_belief, action, observation)

            results.beliefs.append(current_belief)
            results.actions.append(action)
            results.observations.append(observation)
            results.rewards.append(reward)

            if self.visualizer is not None:
                self.visualizer.update_action_obs_reward(results.actions, results.observations, results.rewards)
                belief_im = plot_belief_with_history(current_belief, history=history,
                                                     actual_state=init_block_positions, ret_as_image=True)
                self.visualizer.update_belief_image(belief_im)
                self.visualizer.update_accumulated_reward(accumulated_reward)
                self.visualizer.update_additional_info(f"Step {i + 1}/{self.env.max_steps}")

            if isinstance(observation, ObservationReachedTerminal) or len(current_belief.block_beliefs) == 0 \
                    or observation.steps_left <= 0:
                break

        results.total_reward = accumulated_reward

        print(f"Experiment finished with total reward: {accumulated_reward}. picked up {self.env.n_picked_blocks} blocks")

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
        time.sleep(0.2)
        im, depth = self.env.camera.get_frame_rgb()
        positions, annotations = self.env.position_estimator.get_block_positions_depth(im, depth, help_config,
                                                                                       max_detections=self.env.n_blocks)
        detections_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                             workspace_x_lims_default, workspace_y_lims_default)

        # filter positions that are out of workspace:
        positions = [p for p in positions if workspace_x_lims_default[0]< p[0] < workspace_x_lims_default[1] and
                     workspace_y_lims_default[0] < p[1] < workspace_y_lims_default[1]]

        camera_position = self.env.r1_controller.getActualTCPPose()[:3]
        mus, sigmas = detections_to_distributions(positions, camera_position)

        if len(mus) == 0:
            return [], [], detections_im

        ordered_detection_mus, ordered_detection_sigmas = \
            block_belief.update_from_image_detections_position_distribution(mus, sigmas)

        self.clear_robot1()

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


    def run_from_list_and_save_results(self, row, config_file, results_dir):
        logging.info(f"Belief idx: {row['belief_idx']}, State idx: {row['state_idx']}, Help config idx: {row['help_config_idx_local']}")

        # Prepare experiment data
        init_block_belief = BlocksPositionsBelief(
            self.env.n_blocks,
            self.ws_x_lims,
            self.ws_y_lims,
            init_mus=np.array(eval(row['belief_mus'])),
            init_sigmas=np.array(eval(row['belief_sigmas']))
        )
        init_block_positions = np.array(eval(row['state']))

        if pd.isna(row['help_config']) or row['help_config'] == '' or row['help_config'] == 'None':
            help_config = None
        else:
            help_config = np.array(eval(row['help_config']))

        # Setup experiment directory
        datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = os.path.join(results_dir, datetime_stamp)
        os.makedirs(experiment_dir, exist_ok=True)

        logging.info(f"Running experiment, results will be saved in {experiment_dir}")
        print("Running experiment, results will be saved in", experiment_dir)

        try:
            if self.visualizer:
                self.start_visualizer_if_not_started()
                self.visualizer.update_experiment_type(f"Experiment ID: {row['experiment_id']}")

            if isinstance(self.env.camera, RealsenseCameraWithRecording):
                self.env.camera.start_recording(os.path.join(experiment_dir, "vid"), max_depth=5, fps=20)

            # Run experiment
            results = self.run_single_experiment(
                init_block_positions=init_block_positions,
                init_block_belief=init_block_belief,
                help_config=help_config,
                help_detections_filename=os.path.join(experiment_dir, "help_detections.png") if help_config is not None else None
            )

            # Save results
            results.set_metadata('experiment_id', row['experiment_id'])
            results.set_metadata('belief_idx', row['belief_idx'])
            results.set_metadata('state_idx', row['state_idx'])
            results.set_metadata('help_config_idx_local', row['help_config_idx_local'])
            results.save(os.path.join(experiment_dir, "results.pkl"))

            # Update CSV
            df = pd.read_csv(config_file)
            df['conducted_datetime_stamp'] = df['conducted_datetime_stamp'].astype(str)
            df.loc[df['experiment_id'] == row['experiment_id'], 'conducted_datetime_stamp'] = str(datetime_stamp)
            df.to_csv(config_file, index=False)

            logging.info(f"Experiment ID: {row['experiment_id']} completed and results saved.")

            # Cleanup and save cleanup image
            cleanup_detections = self.clean_up_workspace()
            cv2.imwrite(os.path.join(experiment_dir, "cleanup.png"), cleanup_detections)

        finally:
            if isinstance(self.env.camera, RealsenseCameraWithRecording):
                self.env.camera.stop_recording()

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

        if the camera in the environment is of type RealsenseCameraWithRecording, the method will also save a recording
        of the experiments.

        @param init_block_positions:
        @param init_block_belief:
        @param helper_config:
        @param dirname:
        @return:
        """

        datetime_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"{dirname}/{datetime_stamp}")

        logging.info(f"starting value diff experiment {datetime_stamp}")

        try:
            if self.visualizer is not None:
                self.visualizer.start()
                self.visualizer.update_experiment_type(f"No Help {datetime_stamp}")

            if isinstance(self.env.camera, RealsenseCameraWithRecording):
                self.env.camera.start_recording(f"{dirname}/{datetime_stamp}/vid", max_depth=5, fps=20)

            resutls_no_help = self.run_single_experiment(init_block_positions, init_block_belief, None)
            resutls_no_help.save(f"{dirname}/{datetime_stamp}/no_help.pkl")

            no_help_cleanup_detections = self.clean_up_workspace()
            cv2.imwrite(f"{dirname}/{datetime_stamp}/no_help_cleanup.png", no_help_cleanup_detections)

            if self.visualizer is not None:
                self.visualizer.reset()
                self.visualizer.update_experiment_type(f"With Help {datetime_stamp}")

            results_with_help = self.run_single_experiment(init_block_positions, init_block_belief, helper_config,
                                                           f"{dirname}/{datetime_stamp}/help_detections.png")
            results_with_help.save(f"{dirname}/{datetime_stamp}/with_help.pkl")

            with_help_cleanup_detections = self.clean_up_workspace()
            cv2.imwrite(f"{dirname}/{datetime_stamp}/with_help_cleanup.png", with_help_cleanup_detections)

            logging.info(f"value difference experiment is over, all data saved {datetime_stamp}")
        finally:
            if isinstance(self.env.camera, RealsenseCameraWithRecording):
                self.env.camera.stop_recording()
            if self.visualizer is not None:
                self.visualizer.stop()

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
        init_block_positions = sample_block_positions_from_dists(init_block_belief.block_beliefs,
                                                                 min_dist=0.08)

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
            self.help_configs = np.load("configurations/help_configs.npy")
            logging.debug(f"{len(self.help_configs)} help configs loaded")

        help_config = self.help_configs[np.random.randint(len(self.help_configs))]

        return help_config

    def clear_robot1(self):
        """ make sure robot1 is in a configuration it can never collide with robot2 while it works"""
        self.env.r1_controller.plan_and_moveJ([0.064, -2.455, 0.377, -0.460, -1.161, -0.176])

    def clear_robot2(self):
        """ make sure robot2 is in a configuration it can never collide with robot1 while it works"""
        self.env.r2_controller.plan_and_move_to_xyzrz(goal_tower_position[0], goal_tower_position[1],
                                                      (self.env.n_blocks + 2) * 0.04, 0)

    def safe_distribute_blocks_in_positions(self, block_positions):
        """ distribute blocks from given positions """
        logging.info("distributing blocks from positions")

        self.clear_robot1()

        for bpos in block_positions:
            pile_pos, h = self.piles_manager.pop_next_block()
            start_height = 0.06 + 0.04 * h
            self.env.r2_controller.pick_up(pile_pos[0], pile_pos[1], np.pi/2, start_height)
            self.env.r2_controller.put_down(bpos[0], bpos[1], 0, 0.1)

        self.clear_robot2()

    # def distribute_blocks_from_priors(self, block_positions_distributions: list[BlockPosDist]) -> np.ndarray:
    #     """
    #     sample block positions from distribution, and place actual blocks from the pike at those positions.
    #     returns the sampled block positions
    #     """
    #     logging.info("distributing blocks from priors")
    #
    #     self.clear_robot1()
    #     block_positions = ur5e_2_distribute_blocks_from_block_positions_dists(block_positions_distributions,
    #                                                                           self.env.r2_controller)
    #     self.clear_robot2()
    #
    #     return block_positions

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
        # lookat = [np.mean(self.ws_x_lims), np.mean(self.ws_y_lims), 0]
        # lookat[0] += 0.15
        # lookat[1] += 0.15
        # clean_up_sensor_config = lookat_verangle_horangle_distance_to_robot_config(lookat,
        #                                                                            vertical_angle=60,
        #                                                                            horizontal_angle=30,
        #                                                                            distance=0.7,
        #                                                                            gt=self.env.gt,
        #                                                                            robot_name="ur5e_1")
        clean_up_sensor_config = [1.2647, -1.1831, 1.4136, -2.1864, -2.1700, 0.9569]
        clean_up_sensor_config = to_canonical_config(clean_up_sensor_config)

        self.env.r1_controller.plan_and_moveJ(clean_up_sensor_config)
        im, depth = self.env.camera.get_frame_rgb()
        positions, annotations = self.env.position_estimator.get_block_positions_depth(im, depth,
                                                                                       clean_up_sensor_config,
                                                                                       max_detections=self.env.n_blocks)
        plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                       workspace_x_lims_default, workspace_y_lims_default)
        if self.visualizer is not None:
            self.visualizer.update_detection_image(plot_im, "Last Clean Up Detections")
        self.clear_robot1()
        logging.info(f"detected {len(positions)} blocks")

        # filter positions that are too far from workspace:
        positions = [p for p in positions if workspace_x_lims_default[0] - 0.1 < p[0] < workspace_x_lims_default[1] + 0.1 and
                     workspace_y_lims_default[0] - 0.1 < p[1] < workspace_y_lims_default[1] + 0.1 ]

        # now clean the blocks:
        if put_back_to_stack:
            # TODO: need to intergrate with piles management
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

        return cv2.cvtColor(plot_im, cv2.COLOR_BGR2RGB)

    def start_visualizer_if_not_started(self):
        if self.visualizer is not None and not self.visualizer.running:
            self.visualizer.start()

    def stop_visualizer_if_started(self):
        if self.visualizer is not None and self.visualizer.running:
            self.visualizer.stop()


class BlockPilesManager:
    # first pile is at the corner
    piles_positions = [[-0.3986, -1.5227],
                       [-0.4786, -1.5227]]

    def __init__(self):
        self.piles_current_heights = [4, 4]
        self.piles_max_heights = [4, 4]

    def reset(self):
        """
         call this after piles are really reset! This does not change the actual environment!
        """
        self.piles_current_heights = copy(self.piles_max_heights)

    def pop_next_block(self):
        """ return (position, height) of next pile. height is in blocks and position is xy and world frame"""
        # find next non-empty pile:
        for i, h in enumerate(self.piles_current_heights):
            if h > 0:
                h_before_pikcup = copy(h)
                self.piles_current_heights[i] -= 1
                return self.piles_positions[i], h_before_pikcup

        raise ValueError("all piles are empty")

    def push_back_block(self,):
        # find next pile that is not full
        for i, h in enumerate(self.piles_current_heights):
            if h < self.piles_max_heights[i]:
                self.piles_current_heights[i] += 1
                return

        raise ValueError("all piles are full")


