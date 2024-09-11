import time
from copy import copy

import numpy as np
from scipy.interpolate import interp1d

from .simulation_motion_planner import SimulationMotionPlanner
from ..mujoco_env.voa_world import WorldVoA

FACING_DOWN_R = [[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]]


def compose_transformation_matrix(rotation, translation):
    # Set the upper-left 3x3 part to the rotation matrix
    rotation_flattened = np.array(rotation).flatten()

    # Set the upper-right 3x1 part to the translation vector
    translation = np.array(translation)

    return rotation_flattened, translation


def point_in_square(square_center, edge_length, point):
    # Calculate half the edge length to determine the bounds
    half_edge = edge_length / 2

    # Calculate the left, right, bottom, and top boundaries of the square
    left_bound = square_center[0] - half_edge
    right_bound = square_center[0] + half_edge
    bottom_bound = square_center[1] - half_edge
    top_bound = square_center[1] + half_edge

    # Check if the point is within these boundaries
    return (left_bound <= point[0] <= right_bound) and (bottom_bound <= point[1] <= top_bound)


def canonize_config(config, boundries=(1.3 * np.pi,) * 6):
    for i in range(6):
        while config[i] > boundries[i]:
            config[i] -= 2 * np.pi
        while config[i] < -boundries[i]:
            config[i] += 2 * np.pi
    return config


class MotionExecutor:
    def __init__(self, env: WorldVoA):
        self.env = env
        self.motion_planner = SimulationMotionPlanner()
        self.env.reset()

        self.default_config = [0.0, -1.2, 0.8419, -1.3752, -1.5739, -2.3080]

        state = self.env.get_state()
        self.blocks_positions_dict = state['object_positions']

        # set current configuration
        for robot, pose in state['robots_joint_pos'].items():
            self.motion_planner.update_robot_config(robot, pose)
        # for block_name, pos in blocks_positions_dict.items():
        #     self.motion_planner.add_block(name=block_name, position=pos)

        self.time_step = self.env._mj_model.opt.timestep * self.env.frame_skip

        # self.pid_controllers = {}
        # for robot in self.env.robots_joint_pos.keys():
        #     self.pid_controllers[robot] = [PIDController(kp=1.0, ki=0.1, kd=0.05) for _ in range(6)]

    def moveJ(self, robot_name, target_joints, speed=1.0, acceleration=1.0, tolerance=0.003):
        current_joints = self.env.robots_joint_pos[robot_name]
        current_velocities = self.env.robots_joint_velocities[robot_name]

        # Calculate the joint differences
        joint_diffs = target_joints - current_joints

        # Calculate the time needed for the movement based on max velocity and acceleration
        max_joint_diff = np.max(np.abs(joint_diffs))
        time_to_max_velocity = speed / acceleration
        distance_at_max_velocity = max_joint_diff - 0.5 * acceleration * time_to_max_velocity ** 2

        if distance_at_max_velocity > 0:
            total_time = 2 * time_to_max_velocity + distance_at_max_velocity / speed
        else:
            total_time = 2 * np.sqrt(max_joint_diff / acceleration)

        # Calculate the number of steps based on the frame skip and simulation timestep
        num_steps = int(total_time / self.time_step)
        max_steps = num_steps * 2  # Set a maximum number of steps (2x the expected number)

        # Generate smooth joint trajectories
        trajectories = []
        for i, diff in enumerate(joint_diffs):
            if abs(diff) > tolerance:
                trajectory = self.generate_smooth_trajectory(current_joints[i], target_joints[i], num_steps)
                trajectories.append(trajectory)
            else:
                trajectories.append(np.full(num_steps, current_joints[i]))

        # Execute the trajectory
        for step in range(max_steps):
            if step < num_steps:
                target_positions = [traj[step] for traj in trajectories]
            else:
                target_positions = target_joints

            actions = {robot: self.env.robots_joint_pos[robot] for robot in self.env.robots_joint_pos if
                       robot != robot_name}
            actions[robot_name] = target_positions
            self.env.step(actions)

            # Check if we've reached the target joints
            current_joints = self.env.robots_joint_pos[robot_name]
            if np.allclose(current_joints, target_joints, atol=tolerance):
                break

        # TODO: Handle cases where target joints are unreachable

        if step == max_steps - 1:
            # TODO: Log that the movement timed out
            pass

    def generate_smooth_trajectory(self, start, end, num_steps):
        t = np.linspace(0, 1, num_steps)
        trajectory = start + (end - start) * (3 * t ** 2 - 2 * t ** 3)
        return trajectory

    def moveJ_path(self, robot_name, path_configs, speed=1.0, acceleration=1.0, blend_radius=0.05, tolerance=0.003):
        """
        Move the robot through a path of joint configurations with blending.

        :param robot_name: Name of the robot to move
        :param path_configs: List of joint configurations to move through
        :param speed: Maximum joint speed
        :param acceleration: Maximum joint acceleration
        :param blend_radius: Blend radius for smoothing between configurations
        :param tolerance: Tolerance for considering a point reached
        """
        if len(path_configs) == 1:
            return self.moveJ(robot_name, path_configs[0], speed, acceleration, tolerance)

        path_configs = np.asarray(path_configs)
        full_trajectory = []

        for i in range(len(path_configs) - 1):
            start_config = np.array(path_configs[i])
            end_config = np.array(path_configs[i + 1])

            if i == 0:
                # For the first segment, start from the initial configuration
                segment_trajectory = self.generate_trajectory(start_config, end_config, speed, acceleration,
                                                              blend_radius, blend_start=True,
                                                              blend_end=(i < len(path_configs) - 2))
            elif i == len(path_configs) - 2:
                # For the last segment, end at the final configuration
                segment_trajectory = self.generate_trajectory(start_config, end_config, speed, acceleration,
                                                              blend_radius, blend_start=True, blend_end=False)
            else:
                # For middle segments, blend at both ends
                segment_trajectory = self.generate_trajectory(start_config, end_config, speed, acceleration,
                                                              blend_radius, blend_start=True, blend_end=True)

            full_trajectory.extend(segment_trajectory)

        # Execute the full trajectory
        self.execute_trajectory(robot_name, full_trajectory, tolerance)

    def generate_trajectory(self, start_config, end_config, speed, acceleration, blend_radius, blend_start=True,
                            blend_end=True):
        distance = np.linalg.norm(end_config - start_config)
        total_time = distance / speed
        num_steps = max(int(total_time / self.time_step), 2)  # Ensure at least 2 steps

        trajectory = []

        for t in np.linspace(0, 1, num_steps):
            if blend_start and t < 0.5:
                # Apply blending at the start
                t_blend = t * 2
                config = start_config + (end_config - start_config) * blend_radius * (t_blend ** 2 * (3 - 2 * t_blend))
            elif blend_end and t >= 0.5:
                # Apply blending at the end
                t_blend = (t - 0.5) * 2
                config = start_config + (end_config - start_config) * (
                        1 - blend_radius * ((1 - t_blend) ** 2 * (3 - 2 * (1 - t_blend))))
            else:
                # Linear interpolation in the middle
                config = start_config + (end_config - start_config) * t

            trajectory.append(config)

        return trajectory

    def execute_trajectory(self, robot_name, trajectory, tolerance):
        max_steps = len(trajectory) * 2  # Set a maximum number of steps (2x the expected number)

        for step in range(max_steps):
            if step < len(trajectory):
                target_positions = trajectory[step]
            else:
                target_positions = trajectory[-1]

            actions = {robot: self.env.robots_joint_pos[robot] for robot in self.env.robots_joint_pos if
                       robot != robot_name}
            actions[robot_name] = target_positions
            self.env.step(actions)

            # Check if we've reached the final configuration
            current_joints = self.env.robots_joint_pos[robot_name]
            if np.allclose(current_joints, trajectory[-1], atol=tolerance):
                break

        if step == max_steps - 1:
            print(f"WARNING: Movement for {robot_name} timed out before reaching the final configuration.")

    def plan_and_moveJ(self, robot_name, target_joints, speed=1.0, acceleration=1.0, blend_radius=0.05, tolerance=0.003,
                       max_planning_time=5, max_length_to_distance_ratio=2):
        curr_joint_state = self.env.robots_joint_pos[robot_name]
        path = self.motion_planner.plan_from_start_to_goal_config(robot_name, curr_joint_state, target_joints,
                                                                  max_time=max_planning_time,
                                                                  max_length_to_distance_ratio=max_length_to_distance_ratio)
        self.moveJ_path(robot_name, path, speed, acceleration, blend_radius=blend_radius, tolerance=tolerance)

    def plan_and_move_to_xyz_facing_down(self, robot_name, target_xyz, speed=1.0, acceleration=1.0, blend_radius=0.05,
                                         tolerance=0.003, max_planning_time=5, max_length_to_distance_ratio=2,
                                         cannonized_config=True):
        target_transform = compose_transformation_matrix(FACING_DOWN_R, target_xyz)
        goal_config = self.facing_down_ik(robot_name, target_transform, max_tries=50)

        if cannonized_config:
            goal_config = canonize_config(goal_config)

        self.plan_and_moveJ(robot_name=robot_name,
                            target_joints=goal_config,
                            speed=speed,
                            acceleration=acceleration,
                            blend_radius=blend_radius,
                            tolerance=tolerance,
                            max_planning_time=max_planning_time,
                            max_length_to_distance_ratio=max_length_to_distance_ratio)

    def moveL(self, robot_name, target_position, speed=0.1, acceleration=0.1, tolerance=0.003):
        """
        Move the robot's end-effector in a straight line to the target pose.

        :param robot_name: Name of the robot to move
        :param target_pose: Target pose as a 4x4 transformation matrix
        :param speed: Maximum linear speed of the end-effector (m/s)
        :param acceleration: Maximum linear acceleration of the end-effector (m/s^2)
        :param tolerance: Tolerance for considering the target reached
        """
        current_joints = self.env.robots_joint_pos[robot_name]
        current_pose = self.motion_planner.get_forward_kinematics(robot_name, current_joints)


        # Extract start and end positions
        start_pos = np.asarray(current_pose[1])
        end_pos = target_position

        # Calculate the total distance
        distance = np.linalg.norm(np.asarray(end_pos) - np.asarray(start_pos))

        # Calculate the time needed for the movement
        time_to_max_speed = speed / acceleration
        distance_at_max_speed = distance - acceleration * time_to_max_speed ** 2

        if distance_at_max_speed > 0:
            total_time = 2 * time_to_max_speed + distance_at_max_speed / speed
        else:
            total_time = 2 * np.sqrt(distance / acceleration)

        # Calculate the number of steps
        num_steps = int(total_time / self.time_step)
        max_steps = num_steps * 2  # Set a maximum number of steps

        # Generate the linear trajectory
        trajectory = []
        for t in np.linspace(0, 1, num_steps):
            interpolated_pos = start_pos + (end_pos - start_pos) * t

            # Compute inverse kinematics for the interpolated pose
            target_joints = self.motion_planner.ik_solve(robot_name, (current_pose[0], interpolated_pos),
                                                         start_config=current_joints)
            if target_joints is None or target_joints == []:
                print(f"WARNING: IK solution not found for {robot_name} at step {t}")
                continue

            trajectory.append(target_joints)

        # Execute the trajectory
        for step in range(max_steps):
            if step < len(trajectory):
                target_positions = trajectory[step]
            else:
                target_positions = trajectory[-1]

            actions = {robot: self.env.robots_joint_pos[robot] for robot in self.env.robots_joint_pos if
                       robot != robot_name}
            actions[robot_name] = target_positions
            self.env.step(actions)

            # Check if we've reached the final configuration
            current_joints = self.env.robots_joint_pos[robot_name]
            current_pose = self.motion_planner.get_forward_kinematics(robot_name, current_joints)
            if np.linalg.norm(np.asarray(current_pose[1]) - np.asarray(end_pos)) < tolerance:
                break

        if step == max_steps - 1:
            print(f"WARNING: Linear movement for {robot_name} timed out before reaching the target pose.")

    def reset(self, randomize=True, block_positions=None):
        state = self.env.reset(randomize=randomize, block_positions=block_positions)
        self.blocks_positions_dict = state['object_positions']
        return state

    def activate_grasp(self, wait_steps=5, render_freq=8):
        self.env.set_gripper(True)
        self.motion_planner.attach_box_to_ee()
        return self.wait(wait_steps, render_freq=render_freq)

    def deactivate_grasp(self, wait_steps=5, render_freq=8):
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
            # if i % render_freq == 0:
            # frames.append(self.env.render())

        # account for falling objects
        # self.update_blocks_positions()

        return True, frames, state

    def facing_down_ik(self, agent, target_transform, max_tries=20):
        # Use inverse kinematics to get the joint configuration for this pose
        target_config = self.motion_planner.ik_solve(agent, target_transform)
        if target_config is None:
            target_config = []
        shoulder_constraint_for_down_movement = 0.1

        def valid_shoulder_angle(q):
            return -shoulder_constraint_for_down_movement > q[1] > -np.pi + shoulder_constraint_for_down_movement

        trial = 1
        while (self.motion_planner.is_config_feasible(agent, target_config) is False or
               valid_shoulder_angle(target_config) is False) \
                and trial < max_tries:
            trial += 1
            # try to find another solution, starting from other random configurations:
            q_near = np.random.uniform(-np.pi / 2, np.pi / 2, 6)
            target_config = self.motion_planner.ik_solve(agent, target_transform, start_config=q_near)
            if target_config is None:
                target_config = []

        return target_config

    def check_point_in_block(self, x, y):
        for block_id, pos in self.blocks_positions_dict.items():
            box_center = pos[:2].tolist()
            if point_in_square(square_center=box_center, edge_length=.04, point=[x, y]):
                return block_id
        return None

    def pick_up(self, agent, x, y, start_height=0.15):

        self.plan_and_move_to_xyz_facing_down(agent,
                                              [x, y, start_height],
                                              speed=3.,
                                              acceleration=3.,
                                              blend_radius=0.05,
                                              tolerance=0.1, )
        above_block_config = self.env.robots_joint_pos[agent]

        self.moveL(agent,
                   (x, y, 0.03),
                   speed=0.1,
                   acceleration=0.1,
                   tolerance=0.003)
        self.wait(5)
        _ = self.activate_grasp()
        self.wait(5)
        self.moveJ(agent, above_block_config, speed=3., acceleration=3., tolerance=0.1)

    def put_down(self, agent, x, y, z):
        if self.env.gripper_state_closed is False:
            print('There is no block to put down')
            return False, None, self.env.get_state()
        target_position = [x, y, z]
        target_transform = compose_transformation_matrix(FACING_DOWN_R, target_position)
        target_config = self.facing_down_ik(agent, target_transform)
        success, frames, state = self.move_to_config(agent, target_config)
        if not success:
            return False, frames, state

        grasp_suc, grasp_frames, state = self.deactivate_grasp()

        d_success, d_frames, state = self.move_to_config(agent, self.default_config)

        return grasp_suc and d_success, np.concatenate([frames, grasp_frames, d_frames]), state

    def move_and_detect_height(self, agent, x, y, start_z=0.1, step_size=0.01, force_threshold=10, max_steps=500):
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
        target_transform = compose_transformation_matrix(FACING_DOWN_R, target_position)

        target_config = self.facing_down_ik(agent, target_transform)

        success, frames, state = self.move_to_config(agent, target_config)
        if not success:
            return False, frames, state

        # Now, start moving down until contact is detected
        target_transform[1][2] = 0.06
        total_frames = frames

        sense_config = self.facing_down_ik(agent, target_transform)
        success, new_frames, state = self.move_to_config(agent, sense_config, tolerance=0.001,
                                                         max_steps_per_section=50)
        total_frames.extend(new_frames)

        force_data_z = self.env.get_state()['robots_force'][agent][2]

        success, frames, state = self.move_to_config(agent, self.default_config)

        total_frames.extend(frames)

        return not self.check_point_in_block(x, y) is None, total_frames, state
