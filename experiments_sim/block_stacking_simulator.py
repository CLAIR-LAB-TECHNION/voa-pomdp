import numpy as np
import logging

from modeling.pomdp_problem.domain.action import ActionBase, ActionSense, ActionAttemptStack
from modeling.pomdp_problem.domain.observation import ObservationBase, ObservationReachedTerminal, \
    ObservationSenseResult, ObservationStackAttemptResult
from multi_mujoco.mujoco_env.voa_world import WorldVoA
from multi_mujoco.motion_planning.motion_executor import MotionExecutor, FACING_DOWN_R, canonize_config
from multi_mujoco.mujoco_env.common.ur5e_fk import forward


class BlockStackingSimulator:
    def __init__(self,
                 max_steps=20,
                 stacking_reward=1,
                 finish_ahead_of_time_reward_coeff=0.1,
                 sensing_cost_coeff=0.05,
                 stacking_cost_coeff=0.05,
                 visualize_mp=True,
                 render_sleep_to_maintain_fps=True,
                 ):
        """

        @param max_steps:
        @param stacking_reward:
        @param finish_ahead_of_time_reward_coeff:
        @param sensing_cost_coeff:
        @param stacking_cost_coeff:
        @param visualize_mp:
        @param render_sleep_to_maintain_fps: only relevant when render_mode is 'human'
        """

        self.n_blocks = 4

        self.max_steps = max_steps
        self.stacking_reward = stacking_reward
        self.finish_ahead_of_time_reward_coeff = finish_ahead_of_time_reward_coeff
        self.sensing_cost_coeff = sensing_cost_coeff
        self.stacking_cost_coeff = stacking_cost_coeff

        self.mujoco_env = WorldVoA(render_sleep_to_maintain_fps=render_sleep_to_maintain_fps)
        self.motion_executor = MotionExecutor(env=self.mujoco_env)

        if visualize_mp:
            self.motion_executor.motion_planner.visualize(window_name="sim_motion_planner")

        self.steps = 0
        self.n_picked_blocks = 0
        self.current_robot_position = None

        self.tower_pos = [-0.45, -1.15]

    def reset(self, block_positions):
        logging.info(f"resetting with block_positions: {block_positions}")

        self.clear_r1_no_motion()
        self.clear_r2_no_motion()
        self.mujoco_env.set_block_positions_on_table(block_positions)

        self.steps = 0
        self.n_picked_blocks = 0
        self.current_robot_position = self.get_r2_xy()
        self.motion_executor.wait(1)

    def step(self, action: ActionBase) -> tuple[ObservationBase, float]:
        if self.steps >= self.max_steps:
            print("reached max steps, episode is already done")
            return ObservationReachedTerminal(), 0
        if self.n_picked_blocks == self.n_blocks:
            print("all blocks are already picked, episode is done")
            return ObservationReachedTerminal(), 0

        logging.info(f"performing step {self.steps}, action: {action}")

        self.steps += 1
        steps_left = self.max_steps - self.steps
        reward = 0

        if isinstance(action, ActionSense):
            occupied = self.motion_executor.sense_for_block(agent='ur5e_2',
                                                            x=action.x,
                                                            y=action.y)
            new_robot_pos = self.get_r2_xy()
            observation = ObservationSenseResult(occupied,
                                                 robot_position=new_robot_pos,
                                                 steps_left=steps_left)
            reward -= self.sensing_cost_coeff * \
                      np.linalg.norm(np.array(self.current_robot_position) - np.array(new_robot_pos))

            self.current_robot_position = new_robot_pos

        elif isinstance(action, ActionAttemptStack):
            is_picked = self.motion_executor.pick_up(agent='ur5e_2', x=action.x, y=action.y)
            if is_picked:
                self.n_picked_blocks += 1

                init_tower_height = self.mujoco_env.get_tower_height_at_point(self.tower_pos)
                self.motion_executor.put_down('ur5e_2', *self.tower_pos,)
                final_tower_height = self.mujoco_env.get_tower_height_at_point(self.tower_pos)

                if final_tower_height > init_tower_height:
                    reward += self.stacking_reward

            reward -= self.stacking_cost_coeff * \
                      np.linalg.norm(np.asarray(self.current_robot_position) - np.array((action.x, action.y)))
            reward -= self.stacking_cost_coeff * \
                      np.linalg.norm(np.array([action.x, action.y]) - np.asarray(self.tower_pos))

            if self.n_picked_blocks == self.n_blocks:
                reward += self.finish_ahead_of_time_reward_coeff * steps_left
                steps_left = 0

            self.current_robot_position = self.get_r2_xy()
            observation = ObservationStackAttemptResult(is_picked,
                                                        robot_position=self.current_robot_position,
                                                        steps_left=steps_left)
        else:
            raise ValueError(f"Invalid action type: {action}")

        logging.info(f"performed action, reward: {reward}, observation: {observation}")

        return observation, reward


    def sense_camera_r1(self, robot_config):
        self.motion_executor.plan_and_moveJ("ur5e_1", robot_config)

        actual_config = self.mujoco_env.robots_joint_pos["ur5e_1"]
        camera_pose = self.motion_executor.motion_planner.get_forward_kinematics("ur5e_1", actual_config)
        position = np.array(camera_pose[1])
        orientation = np.array(camera_pose[0]).reshape(3, 3)
        return self.mujoco_env.render_image_from_pose(position, orientation)

    def move_r2_above_tower(self):
        self.motion_executor.plan_and_move_to_xyz_facing_down("ur5e_2",
                                                              (self.tower_pos[0], self.tower_pos[1], 0.3),
                                                              speed=3,
                                                              acceleration=3,
                                                              tolerance=0.05)

    def get_r2_xy(self):
        robot_j = self.mujoco_env.robots_joint_pos["ur5e_2"]
        ee_transform = self.motion_executor.motion_planner.get_forward_kinematics("ur5e_2", robot_j)
        return ee_transform[1][:2]

    def clear_r1_no_motion(self):
        self.mujoco_env.set_robot_joints("ur5e_1", [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, 0, 0])

    def clear_r2_no_motion(self):
        pose = [np.array(FACING_DOWN_R).flatten(), [self.tower_pos[0], self.tower_pos[1], 0.3]]
        config = self.motion_executor.facing_down_ik("ur5e_2", pose)
        config = canonize_config(config)
        self.mujoco_env.set_robot_joints("ur5e_2", config)
