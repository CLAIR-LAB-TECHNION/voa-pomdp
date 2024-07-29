import numpy as np
from klampt import vis

from .simulation_motion_planner import SimulationMotionPlanner
from ..mujoco_env.voa_world import WorldVoA

FACING_DOWN_R = [[0, 0, -1],
                 [0, 1, 0],
                 [1, 0, 0]]


def compose_transformation_matrix(rotation, translation):
    # Set the upper-left 3x3 part to the rotation matrix
    rotation_flattened = np.array(rotation).flatten()

    # Set the upper-right 3x1 part to the translation vector
    translation = np.array(translation)

    return rotation_flattened, translation


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
        return self.env.get_state()

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
        state = None
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

        return True, frames, state

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
        state = None
        for config in path:
            success, frames_curr, state = self.move_to(
                agent,
                config,
                tolerance=tolerance,
                end_vel=end_vel,
                max_steps=max_steps_per_section,
                render_freq=render_freq)
            frames.extend(frames_curr)
            if not success:
                return False, frames

        return True, frames, state

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
        success, frames, state = self.execute_path(
            agent,
            path,
            tolerance=tolerance,
            end_vel=end_vel,
            max_steps_per_section=max_steps_per_section,
            render_freq=render_freq)

        # after executing a motion, blocks position can change, update the motion planner:
        self.update_blocks_positions()

        return success, frames, state

    def move_to_pose(self, agent, target_position, target_orientation):
        target_transform = compose_transformation_matrix(target_orientation, target_position)
        target_config = self.motion_planner.ik_solve(agent, target_transform)
        if target_config is None:
            print(f'({target_config}) is unreachable')
            return False, None, None

        return self.move_to_config(agent, target_config)

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
        state = None

        # move to current position, i.e., stay in place
        maintain_pos = self.env.robots_joint_pos
        for i in range(n_steps):
            state = self.env.step(maintain_pos)
            if i % render_freq == 0:
                frames.append(self.env.render())

        # account for falling objects
        self.update_blocks_positions()

        return True, frames, state

    def move_and_detect_height(self, agent, x, y, start_z=0.2, step_size=0.01, force_threshold=0.1, max_steps=500):
        """
        Move the robot above a specified (x,y) point and lower it until contact is detected.

        @param agent: agent id to move
        @param x: x-coordinate to move above
        @param y: y-coordinate to move above
        @param start_z: starting z-coordinate (height above the surface)
        @param step_size: how much to lower the end-effector in each step
        @param force_threshold: force threshold to detect contact
        @param max_steps: maximum number of steps to take before stopping
        @return: success, frames, final_height
        """
        # First, move to the position above (x, y, start_z)
        target_position = [x, y, start_z]
        target_orientation = FACING_DOWN_R  # Assuming this is defined in your class
        target_transform = compose_transformation_matrix(target_orientation, target_position)

        vis.add('world', self.motion_planner.world)

        # Use inverse kinematics to get the joint configuration for this pose
        target_config = self.motion_planner.ik_solve(agent, target_transform)
        if target_config is None:
            print(f'({x}, {y}, {start_z}) is unreachable')
            return False, None, None

        success, frames, state = self.move_to_config(agent, target_config)
        if not success:
            return False, frames, None

        # Now, start moving down until contact is detected
        current_z = start_z
        total_frames = frames

        for _ in range(max_steps):
            current_z -= step_size
            target_transform[1][2] = current_z

            target_config = self.motion_planner.ik_solve(agent, target_transform)
            success, new_frames, state = self.move_to_config(agent, target_config, tolerance=0.001,
                                                             max_steps_per_section=50)
            total_frames.extend(new_frames)

            if not success:
                return False, total_frames, None

            # Check force sensor data
            force_data = self.env.get_state()[agent]['sensor']
            if abs(force_data[2]) > force_threshold:  # Assuming z-axis is the third component
                print(f"Contact detected at height: {current_z}")
                return True, total_frames, current_z

        print("Max steps reached without detecting contact")
        return False, total_frames, current_z
