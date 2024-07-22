""" a wrapper around spear env to simplify and fix some issues with the environment """
from copy import deepcopy
from collections import namedtuple

from mujoco_env import MujocoEnv

from .world_utils.object_manager import ObjectManager
from .world_utils.grasp_manager import GraspManager
from .world_utils.configurations_and_constants import *


class WorldVoA():
    def __init__(self, render_mode='human', cfg=env_cfg):
        self.render_mode = render_mode
        self._env = MujocoEnv.from_cfg(cfg=cfg, render_mode=render_mode, frame_skip=frame_skip)
        obs, info = self._env.reset()  # once, for info, later again
        self._mj_model = info['privileged']['model']
        self._mj_data = info['privileged']['data']
        self._env_entities = {name: agent.entity for name, agent in self._env.agents.items()}
        self.robots_joint_pos = {}
        self.robots_joint_velocities = {}
        for agent in self._env_entities.keys():
            self.robots_joint_pos[agent] = np.zeros((1, 6))  # will be updated in reset
            self.robots_joint_velocities[agent] = np.zeros((1, 6))  # --""--
        self.gripper_state_closed = False  # --""--
        self.max_joint_velocities = INIT_MAX_VELOCITY

        self._object_manager = ObjectManager(self._mj_model, self._mj_data)
        self._grasp_manager = GraspManager(self._mj_model, self._mj_data, self._object_manager, min_grasp_distance=0.1)

        self._ee_mj_data = self._mj_data.body('rethink_mount_stationary_1/robot_1_ur5e/robot_1_adhesive gripper/')
        # dt = self._mj_model.opt.timestep * frame_skip
        # self._pid_controller = PIDController(kp, ki, kd, dt)

        self.reset()

    def reset(self):
        self.max_joint_velocities = INIT_MAX_VELOCITY

        obs, _ = self._env.reset()
        for agent in obs.keys():
            self.robots_joint_pos[agent] = obs[agent]['robot_state'][:6]
            self.robots_joint_velocities[agent] = obs[agent]["robot_state"][6:12]
        self.gripper_state_closed = False
        self._grasp_manager.release_object()
        self._object_manager.reset_object_positions()

        self.step(self.robots_joint_pos, gripper_closed=False)

        if self.render_mode == "human":
            self._env.render()

        return self.get_state()

    def step(self, target_joint_pos, gripper_closed=None):
        # if reset_pid:
        #     self._pid_controller.reset_endpoint(target_joint_pos)
        if gripper_closed is None:
            gripper_closed = self.gripper_state_closed

        self._env_step(target_joint_pos, gripper_closed)
        self._clip_joint_velocities()

        if gripper_closed:
            if self._grasp_manager.attatched_object_name is not None:
                self._grasp_manager.update_grasped_object_pose()
            else:
                self._grasp_manager.grasp_nearest_object_if_close_enough()
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

    def close(self):
        self._env.close()

    def get_state(self):
        object_positions = self._object_manager.get_all_object_positons_dict()
        state = {"robots_joint_pos": self.robots_joint_pos,
                 "robots_joint_velocities": self.robots_joint_velocities,
                 "gripper_state_closed": self.gripper_state_closed,
                 "object_positions": object_positions,
                 "grasped_object": self._grasp_manager.attached_object_name,
                 "geom_contact": convert_mj_struct_to_namedtuple(self._env.sim.data.contact)}

        return deepcopy(state)

    def get_object_pos(self, name: str):
        return self._object_manager.get_object_pos(name)

    def move_to(self, agent, target_joint_pos, tolerance=0.05, end_vel=0.1, max_steps=None):
        """
        move robot joints to target config, until it is close within tolerance, or max_steps exceeded.
        @param agent: agent id
        @param target_joint_pos: position to move to
        @param tolerance: distance withing configuration space to target to consider as reached
        @param max_steps: maximum steps to take before stopping
        @return: state, success

        success is true if reached goal within tolerance, false otherwise
        """
        # self._pid_controller.reset_endpoint(target_joint_pos)

        step = 0
        actions = {}
        for other_agents, pos in self.robots_joint_pos.items():
            actions[other_agents] = pos
        actions[agent] = target_joint_pos
        while np.linalg.norm(self.robots_joint_pos[agent] - target_joint_pos) > tolerance \
                or np.linalg.norm(self.robots_joint_velocities[agent]) > end_vel:
            if max_steps is not None and step > max_steps:
                return self.get_state(), False

            self.step(actions, self.gripper_state_closed)
            step += 1

        return self.get_state(), True

    def set_gripper(self, closed: bool):
        """
        close/open gripper and don't change robot configuration
        @param closed: true if gripper should be closed, false otherwise
        @return: None
        """
        self._env_step(self.robots_joint_pos, closed)

    def _clip_joint_velocities(self):
        new_vel = self.robots_joint_velocities.copy()
        for agent, vel in new_vel.items():
            new_vel[agent] = np.clip(vel, -self.max_joint_velocities, self.max_joint_velocities)
            self._env_entities[agent].set_state(velocity=new_vel[agent])
        self.robots_joint_velocities = new_vel

    def _env_step(self, target_joint_pos, gripper_closed):
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
        self.gripper_state_closed = gripper_closed

    def get_ee_pos(self):
        return deepcopy(self._ee_mj_data.xpos)


def convert_mj_struct_to_namedtuple(mj_struct):
    """
    convert a mujoco struct to a dictionary
    """
    attrs = [attr for attr in dir(mj_struct) if not attr.startswith('__') and not callable(getattr(mj_struct, attr))]
    return namedtuple(mj_struct.__class__.__name__, attrs)(**{attr: getattr(mj_struct, attr) for attr in attrs})
