import numpy as np

from .simulation_motion_planner import SimulationMotionPlanner
from ..mujoco_env.voa_world import WorldVoA

FACING_DOWN_R = [[0, 0, -1],
                 [0, 1, 0],
                 [1, 0, 0]]


class MotionExecutor:
    def __init__(self, env: WorldVoA):
        self.env = env
        self.motion_planner = SimulationMotionPlanner()
        self.env.reset()

        state = self.env.get_state()
        blocks_positions_dict = state['object_positions']

        # set current configuration
        for robot, pose in state['robots_joint_pos'].items():
            self.motion_planner.update_robot_config(robot, pose)
        for block_name, pos in blocks_positions_dict.items():
            self.motion_planner.add_block(name=block_name, position=pos)

    def reset(self, randomize=True, block_positions=None):
        self.env.reset(randomize=randomize, block_positions=block_positions)
        self.update_blocks_positions()

    def move_to(self, agent, target_config, tolerance=0.05, end_vel=0.1, max_steps=None,
                render_freq=8):
        """
        move robot joints to target config, until it is close within tolerance,
        or max_steps exceeded.
        @param agent: agent id to move
        @param target_config: position to move to
        @param tolerance: distance withing configuration space to target to consider
         as reached
        @param end_vel: end_vel
        @param max_steps: maximum steps to take before stopping, no limit if None
        @param render_freq: how often to render and append a frame
        @return: success, frames
        """
        joint_positions = self.env.robots_joint_pos
        joint_velocities = self.env.robots_joint_velocities

        frames = []

        actions = {}
        for other_agent, config in joint_positions.items():
            actions[other_agent] = config
        actions[agent] = target_config

        i = 0
        while np.linalg.norm(joint_positions[agent] - target_config) > tolerance \
                or np.linalg.norm(joint_velocities[agent]) > end_vel:
            if max_steps is not None and i > max_steps:
                return False, frames

            state = self.env.step(actions)
            joint_positions = state['robots_joint_pos']
            joint_velocities = state['robots_joint_velocities']

            if i % render_freq == 0:
                frames.append(self.env.render())

            i += 1

        return True, frames

    def update_blocks_positions(self):
        blocks_positions_dict = self.env.get_state()['object_positions']
        for name, pos in blocks_positions_dict.items():
            self.motion_planner.move_block(name, pos)

    def execute_path(self, agent, path, tolerance=0.05, end_vel=0.1,
                     max_steps_per_section=200, render_freq=8):
        """
        execute a path of joint positions
        @param agent: agent id to move
        @param path: list of joint positions to follow
        @param tolerance: distance withing configuration space to target to consider as reached to each point
        @param end_vel: maximum velocity to consider as reached to each point
        @param max_steps_per_section: maximum steps to take before stopping at each section
        @param render_freq: how often to render and append a frame
        @return: success, frames
        """
        frames = []
        for config in path:
            success, frames_curr = self.move_to(
                agent,
                config,
                tolerance=tolerance,
                end_vel=end_vel,
                max_steps=max_steps_per_section,
                render_freq=render_freq)
            frames.extend(frames_curr)
            if not success:
                return False, frames

        return True, frames

    def move_to_config(self, agent, target_config, tolerance=0.05, end_vel=0.1, max_steps_per_section=400,
                       render_freq=8):
        """
        move robot to target position and orientation, and update the motion planner
        with the new state of the blocks
        @param agent: agent id to move
        @param target_config: target configuration to move to
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps_per_section: maximum steps to take before stopping a section
        @param render_freq: how often to render and append a frame
        @return: success, frames
        """
        joint_state = self.env.robots_joint_pos[agent]
        path = self.motion_planner.plan_from_start_to_goal_config(agent, joint_state, target_config)
        success, frames = self.execute_path(
            agent,
            path,
            tolerance=tolerance,
            end_vel=end_vel,
            max_steps_per_section=max_steps_per_section,
            render_freq=render_freq)

        # after executing a motion, blocks position can change, update the motion planner:
        self.update_blocks_positions()

        return success, frames

    def activate_grasp(self, wait_steps=10, render_freq=8):
        self.env.set_gripper(True)
        self.motion_planner.attach_box_to_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def deactivate_grasp(self, wait_steps=10, render_freq=8):
        self.env.set_gripper(False)
        self.motion_planner.detach_box_from_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def wait(self, n_steps, render_freq=8):
        frames = []

        # move to current position, i.e., stay in place
        maintain_pos = self.env.robots_joint_pos
        for i in range(n_steps):
            self.env.step(maintain_pos)
            if i % render_freq == 0:
                frames.append(self.env.render())

        # account for falling objects
        self.update_blocks_positions()

        return True, frames
