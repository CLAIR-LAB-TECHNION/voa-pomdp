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

        # self.pid_controllers = {}
        # for robot in self.env.robots_joint_pos.keys():
        #     self.pid_controllers[robot] = [PIDController(kp=1.0, ki=0.1, kd=0.05) for _ in range(6)]

    def reset(self, randomize=True, block_positions=None):
        state = self.env.reset(randomize=randomize, block_positions=block_positions)
        self.blocks_positions_dict = state['object_positions']
        return state

    def move_to(self, agent, target_config, velocity=1.05, acceleration=1.4, tolerance=0.05, end_vel=0.1,
                max_steps=None, render_freq=8):
        joint_positions = self.env.robots_joint_pos
        joint_velocities = self.env.robots_joint_velocities
        frames = []
        actions = {other_agent: config for other_agent, config in joint_positions.items()}

        start_config = joint_positions[agent]
        distance = np.linalg.norm(target_config - start_config)

        # Calculate time needed for acceleration and deceleration
        t_acc = velocity / acceleration
        d_acc = 0.5 * acceleration * t_acc ** 2

        if 2 * d_acc < distance:
            # Trapezoidal profile (with constant velocity phase)
            t_const = (distance - 2 * d_acc) / velocity
            total_time = 2 * t_acc + t_const
        else:
            # Triangular profile (no constant velocity phase)
            t_acc = np.sqrt(distance / acceleration)
            total_time = 2 * t_acc
            velocity = acceleration * t_acc

        # Generate trajectory
        dt = self.env._mj_model.opt.timestep * self.env.frame_skip
        t = np.arange(0, total_time + dt, dt)
        trajectory = self._generate_trajectory(start_config, target_config, t, velocity, acceleration)

        i = 0
        state = None
        while np.linalg.norm(joint_positions[agent] - target_config) > tolerance \
                or np.linalg.norm(joint_velocities[agent]) > end_vel:
            if max_steps is not None and i >= max_steps:
                return False, frames, self.env.get_state()

            current_t = min(i * dt, total_time)
            index = int(current_t / dt)
            desired_pos = trajectory[index]

            actions[agent] = desired_pos

            state = self.env.step(actions)
            joint_positions = state['robots_joint_pos']
            joint_velocities = state['robots_joint_velocities']

            # if i % render_freq == 0:
            #     frames.append(self.env.render())
            i += 1

        self.motion_planner.update_robot_config(agent, joint_positions[agent])
        return True, frames, state

    def _generate_trajectory(self, start, end, t, velocity, acceleration):
        distance = np.linalg.norm(end - start)
        direction = (end - start) / distance if distance > 0 else np.zeros_like(start)

        def position(time):
            if time < velocity / acceleration:
                # Acceleration phase
                s = 0.5 * acceleration * time ** 2
            elif time < t[-1] - velocity / acceleration:
                # Constant velocity phase
                s = 0.5 * velocity ** 2 / acceleration + velocity * (time - velocity / acceleration)
            else:
                # Deceleration phase
                t_dec = t[-1] - time
                s = distance - 0.5 * acceleration * t_dec ** 2

            return start + direction * s

        positions = np.array([position(time) for time in t])
        return positions

    def execute_path(self, agent, path, blending_radius=0.1, velocity=1.05, acceleration=1.4,
                     tolerance=0.005, end_vel=0.1, max_steps_per_section=200, render_freq=8):
        """
        Execute a path of joint positions with blending between waypoints

        @param agent: agent id to move
        @param path: list of joint positions to follow
        @param blending_radius: radius for blending between waypoints
        @param velocity: maximum joint velocity (rad/s)
        @param acceleration: joint acceleration (rad/s^2)
        @param tolerance: distance within configuration space to target to consider as reached to each point
        @param end_vel: maximum velocity to consider as reached to each point
        @param max_steps_per_section: maximum steps to take before stopping at each section
        @param render_freq: how often to render and append a frame
        @return: success, frames, final state
        """
        frames = []
        state = None

        blended_path = self._blend_path(path, blending_radius) if blending_radius > 0 else path

        for config in blended_path:
            success, frames_curr, state = self.move_to(
                agent,
                config,
                velocity=velocity,
                acceleration=acceleration,
                tolerance=0.05,
                end_vel=end_vel,
                max_steps=max_steps_per_section,
                render_freq=render_freq)
            frames.extend(frames_curr)
            # if not success:
            #     return False, frames, self.env.get_state()
        return True, frames, state

    def _blend_path(self, path, blending_radius):
        blended_path = []
        for i in range(len(path)):
            if i == 0 or i == len(path) - 1:
                blended_path.append(path[i])
            else:
                prev = np.asarray(path[i - 1])
                curr = np.asarray(path[i])
                next = np.asarray(path[i + 1])

                # Calculate blending points
                to_prev = prev - curr
                to_next = next - curr
                to_prev_norm = to_prev / np.linalg.norm(to_prev)
                to_next_norm = to_next / np.linalg.norm(to_next)

                blend_start = curr + to_prev_norm * blending_radius
                blend_end = curr + to_next_norm * blending_radius

                blended_path.append(blend_start)
                blended_path.append(blend_end)

        return blended_path

    def moveL_relative(self, agent, direction, distance, velocity=0.1, acceleration=0.1,
                       tolerance=0.001, end_vel=0.01, max_steps=1000, render_freq=8):
        """
        Move the robot's end-effector along a straight line in a specified Cartesian direction.

        @param agent: agent id to move
        @param direction: 3D vector specifying the direction of movement [x, y, z]
        @param distance: distance to move along the specified direction
        @param velocity: maximum linear velocity (m/s)
        @param acceleration: linear acceleration (m/s^2)
        @param tolerance: distance tolerance to consider target reached
        @param end_vel: end velocity tolerance
        @param max_steps: maximum steps to take before stopping
        @param render_freq: how often to render and append a frame
        @return: success, frames, final state
        """
        direction = np.array(direction) / np.linalg.norm(direction)
        current_config = copy(self.env.robots_joint_pos[agent])
        current_ee_transform = self.motion_planner.get_forward_kinematics(agent, current_config)
        current_position = current_ee_transform[1]
        current_orientation = current_ee_transform[0]
        target_position = current_position + direction * distance

        cartesian_path = np.linspace(current_position, target_position, 10)
        joint_path = [self.motion_planner.ik_solve(agent, (current_orientation, pos)) for pos in cartesian_path]

        success, frames, state = self.execute_path(agent,
                                                   joint_path,
                                                   velocity=velocity,
                                                   acceleration=acceleration,
                                                   blending_radius=0.01,
                                                   tolerance=tolerance,
                                                   end_vel=end_vel,
                                                   max_steps_per_section=max_steps,
                                                   render_freq=render_freq)

        return success, frames, state

    def update_blocks_positions(self):
        blocks_positions_dict = self.env.get_state()['object_positions']
        for name, pos in blocks_positions_dict.items():
            self.motion_planner.move_block(name, pos)

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
            velocity=200,
            acceleration=200,
            tolerance=tolerance,
            end_vel=end_vel,
            max_steps_per_section=max_steps_per_section,
            render_freq=render_freq)

        # after executing a motion, blocks position can change, update the motion planner:
        # self.update_blocks_positions()

        return success, frames, state

    def move_to_pose(self, agent, target_position, target_orientation):
        target_transform = compose_transformation_matrix(target_orientation, target_position)
        target_config = self.motion_planner.ik_solve(agent, target_transform)
        if target_config is None:
            print(f'({target_config}) is unreachable')
            return False, None, None

        return self.move_to_config(agent, target_config)

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

    def pick_up(self, agent, x, y, start_height=0.15):
        # move above position:
        target_position = [x, y, start_height]
        target_transform = compose_transformation_matrix(FACING_DOWN_R, target_position)
        target_config = self.facing_down_ik(agent, target_transform)
        success, frames, state = self.move_to_config(agent, target_config)
        if not success:
            return False, frames, state

        success, frames, state = self.moveL_relative(agent, [0, 0, -1], 0.1, velocity=10, acceleration=10,)

        grasp_suc, grasp_frames, _ = self.activate_grasp()

        d_success, d_frames, state = self.move_to_config(agent, self.default_config)

        return grasp_suc and d_success, np.concatenate([frames, grasp_frames, d_frames]), state

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

    def facing_down_ik(self, agent, target_transform, max_tries=20):
        # Use inverse kinematics to get the joint configuration for this pose
        target_config = self.motion_planner.ik_solve(agent, target_transform)
        if target_config is None:
            target_config = []
        shoulder_constraint_for_down_movement = 0.1

        def valid_shoulder_angle(q):
            return -shoulder_constraint_for_down_movement > q[1] > -np.pi + shoulder_constraint_for_down_movement

        trial = 1
        while ((self.motion_planner.is_config_feasible(agent, target_config) is False or valid_shoulder_angle(
                target_config) is False)
               and trial < max_tries):
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
