""" a wrapper around spear env to simplify and fix some issues with the environment """
from copy import deepcopy
from collections import namedtuple

import mujoco as mj
from mujoco import MjvCamera

from .mujoco_env import MujocoEnv

from .world_utils.object_manager import ObjectManager
from .world_utils.grasp_manager import GraspManager
from .world_utils.configurations_and_constants import *


class WorldVoA:
    def __init__(self, render_mode='human', cfg=muj_env_config):
        self.render_mode = render_mode
        self._env = MujocoEnv.from_cfg(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)
        self.frame_skip = frame_skip
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']
        self._env_entities = {name: agent.entity for name, agent in self._env.agents.items()}
        self.robots_joint_pos = {}
        self.robots_joint_velocities = {}
        self.robots_force = {}
        self.robots_camera = {}
        for agent in self._env_entities.keys():
            self.robots_joint_pos[agent] = np.zeros((1, 6))  # will be updated in reset
            self.robots_joint_velocities[agent] = np.zeros((1, 6))  # --""--
            self.robots_force[agent] = 0.0
            self.robots_camera[agent] = []
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY

        self._object_manager = ObjectManager(self._mj_model, self._mj_data)
        self._grasp_manager = GraspManager(self._mj_model, self._mj_data, self._object_manager, min_grasp_distance=0.1)

        self.num_blocks = len(self._object_manager.object_names)

        self.renderer = mj.Renderer(self._mj_model, 480, 480)

        self.camera = MjvCamera()
        self.camera.type = mj.mjtCamera.mjCAMERA_FREE

        self._ee_mj_data = self._mj_data.body('robot_1_ur5e/robot_1_adhesive gripper/')
        # self.dt = self._mj_model.opt.timestep * frame_skip
        # self._pid_controller = PIDController(kp, ki, kd, dt)

        self.reset()

    def close(self):
        self._env.close()

    def reset(self, randomize=True, block_positions=None):
        self.max_joint_velocities = INIT_MAX_VELOCITY

        obs, _ = self._env.reset()
        agents = obs.keys()

        for agent in agents:
            self._env_entities[agent].set_state(position=[-np.pi/2, -np.pi/2, 0, -np.pi/2, 0, 0])

        for agent in agents:
            self.robots_joint_pos[agent] = obs[agent]['robot_state'][:6]
            self.robots_joint_velocities[agent] = obs[agent]["robot_state"][6:12]
            # self.robots_force[agent] = obs[agent]['sensor']
            self.robots_camera[agent] = [obs[agent]['camera'], obs[agent]['camera_pose']]
        self.gripper_state_closed = False
        self._grasp_manager.release_object()
        self._object_manager.reset(randomize=randomize, block_positions=block_positions)

        self.step(self.robots_joint_pos, gripper_closed=False)

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def step(self, target_joint_pos, gripper_closed=None):
        # if reset_pid:
        #     self._pid_controller.reset_endpoint(target_joint_pos)
        if gripper_closed is None:
            gripper_closed = self.gripper_state_closed
        self.gripper_state_closed = gripper_closed

        self._env_step(target_joint_pos)
        self._clip_joint_velocities()

        if gripper_closed:
            if self._grasp_manager.attached_object_name is not None:
                self._grasp_manager.update_grasped_object_pose()
            else:
                self._grasp_manager.grasp_block_if_close_enough()
        else:
            self._grasp_manager.release_object()

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def simulate_steps(self, n_steps):
        """
        simulate n_steps in the environment without moving the robot
        """
        config = self.robots_joint_pos
        for _ in range(n_steps):
            self.step(config)

    def render(self):
        return self._env.render()

    def get_state(self):
        # object_positions = self._object_manager.get_all_block_positions_dict()
        state = {"robots_joint_pos": self.robots_joint_pos,
                 "robots_joint_velocities": self.robots_joint_velocities,
                 # "robots_force": self.robots_force,
                 # "robots_camera": self.robots_camera,
                 "gripper_state_closed": self.gripper_state_closed,
                 # "object_positions": object_positions,
                 "grasped_object": self._grasp_manager.attached_object_name,}
                 # "geom_contact": convert_mj_struct_to_namedtuple(self._env.sim.data.contact)}

        return deepcopy(state)

    def get_tower_height_at_point(self, point):
        block_positions = self._object_manager.get_all_block_positions_dict()
        not_grasped_block_positions = {name: pos for name, pos in block_positions.items()
                                       if name != self._grasp_manager.attached_object_name}
        not_grasped_block_positions = np.array(list(not_grasped_block_positions.values()))

        blocks_near_point = not_grasped_block_positions[
            np.linalg.norm(not_grasped_block_positions[:, :2] - point, axis=1) < 0.03]
        highest_block = np.max(blocks_near_point[:, 2]) if blocks_near_point.size > 0 else 0
        return highest_block

    def get_block_positions(self):
        return list(self._object_manager.get_all_block_positions_dict().values())

    def set_gripper(self, closed: bool):
        """
        close/open gripper and don't change robot configuration
        @param closed: true if gripper should be closed, false otherwise
        @return: None
        """
        self.step(self.robots_joint_pos, closed)

    def _clip_joint_velocities(self):
        # new_vel = self.robots_joint_velocities.copy()
        # for agent, vel in new_vel.items():
        #     new_vel[agent] = np.clip(vel, -self.max_joint_velocities, self.max_joint_velocities)
        #     self._env_entities[agent].set_state(velocity=new_vel[agent])
        # self.robots_joint_velocities = new_vel
        return

    def _env_step(self, target_joint_pos):
        """ run environment step and update state of self accordingly"""

        # joint_control = self._pid_controller.control(self.robot_joint_pos)  # would be relevant if we change to force
        # control and use PID controller, but we are using position control right now.

        # gripper control for the environment, which is the last element in the control vector is completely
        # ignored right now, instead we attach the nearest graspable object to the end effector and maintain it
        # with the grasp manager, outside the scope of this method.

        # action = np.concatenate((target_joint_pos, [int(gripper_closed)])) # would have been if grasping worked
        actions = {}
        for agent, action in target_joint_pos.items():
            actions[agent] = np.concatenate((action, [0]))

        obs, r, term, trunc, info = self._env.step(actions)
        for agent, ob in obs.items():
            self.robots_joint_pos[agent] = ob['robot_state'][:6]
            self.robots_joint_velocities[agent] = ob['robot_state'][6:12]
            # self.robots_force[agent] = obs[agent]['sensor']
            self.robots_camera[agent] = [obs[agent]['camera'], obs[agent]['camera_pose']]

    def get_ee_pos(self):
        return deepcopy(self._ee_mj_data.xpos)

    def render_image_from_pose(self, position, rotation_matrix):

        look_at, distance, elevation_deg, azimuth_deg = pose_to_plane_intersection(position, rotation_matrix)
        self.camera.lookat = look_at
        self.camera.distance = distance
        self.camera.elevation = -elevation_deg
        self.camera.azimuth = -azimuth_deg

        self.renderer.update_scene(self._mj_data, self.camera)

        return self.renderer.render()


def convert_mj_struct_to_namedtuple(mj_struct):
    """
    convert a mujoco struct to a dictionary
    """
    attrs = [attr for attr in dir(mj_struct) if not attr.startswith('__') and not callable(getattr(mj_struct, attr))]
    return namedtuple(mj_struct.__class__.__name__, attrs)(**{attr: getattr(mj_struct, attr) for attr in attrs})


def pose_to_plane_intersection(position, rotation_matrix):
    """
    Calculate intersection point, distance, elevation, and azimuth from a 3D pose to the xy-plane.

    :param position: 3D position vector [x, y, z]
    :param rotation_matrix: 3x3 rotation matrix
    :return: tuple (intersection_point, distance, elevation_deg, azimuth_deg)
    """
    direction = rotation_matrix[:, 2]

    if direction[2] == 0:
        raise ValueError("Direction is parallel to xy-plane, no intersection.")

    t = -position[2] / direction[2]

    intersection_point = position + t * direction
    intersection_point[2] = 0  # Ensure z-coordinate is exactly 0

    distance = np.linalg.norm(position - intersection_point)

    elevation = np.arcsin((position[2] - intersection_point[2]) / distance)
    elevation_deg = np.degrees(elevation)

    delta_y = position[1] - intersection_point[1]
    delta_x = position[0] - intersection_point[0]
    azimuth = np.arctan2(delta_y, delta_x)
    azimuth_deg = np.degrees(azimuth)

    azimuth_deg = (azimuth_deg + 360) % 360

    return intersection_point, distance, elevation_deg, azimuth_deg
