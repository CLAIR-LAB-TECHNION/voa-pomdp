from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_env import mujoco_env
from mujoco_env.tasks.null_task import NullTask

from mujoco_env.common.ur5e_fk import forward
from mujoco_env.common.transform import Transform

# Define the configuration for multiple robots and tasks
cfg = dict(
    scene=dict(
        resource='tableworld',
        render_camera='top-right'
    ),
    robots=dict(
        robot_0=dict(
            resource='ur5e',
            mount='rethink_stationary',
            attachments=['adhesive_gripper'],
            base_pos=[0.5, 0.5, 0],  # Example base position
            base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
        ),
        robot_1=dict(
            resource='ur5e',
            mount='rethink_stationary',
            attachments=['adhesive_gripper'],
            base_pos=[-0.5, -0.5, 0],  # Example base position
            base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
        ),
    ),
    tasks=dict(
        robot_0=NullTask,
        robot_1=NullTask,
    ),
)

camera_pos = np.array([0, 0.5, 0.5])
camera_quat = np.array([0.7071, 0.7071, 0, 0])
camera_rot = R.from_quat(camera_quat).as_matrix()
ee_to_camera = Transform(rotation=camera_rot, translation=camera_pos, from_frame='end_effector', to_frame='camera')

# Initialize the environment with the multi-agent configuration
env = mujoco_env.MujocoEnv.from_cfg(cfg=cfg, render_mode="human", frame_skip=5)

N_EPISODES = 1
N_STEPS = 2000

try:
    for _ in range(N_EPISODES):
        obs, info = env.reset()
        env.render()
        done = {agent_name: False for agent_name in env.agents.keys()}
        i = 0
        while not all(done.values()) and i < N_STEPS:
            i += 1
            actions = env.action_space.sample()

            obs, rewards, term, trunc, info = env.step(actions)
            done = {agent_name: term[agent_name] for agent_name in env.agents.keys()}
            env.render()
            if i == 10:
                img = obs['robot_0']['camera'][:, :, :3].astype(np.uint8)
                world_to_ee = forward(obs['robot_0']['robot_state'][:6])
                world_to_camera = world_to_ee.compose(ee_to_camera).to_pose_quaternion()
                # print(world_to_camera)
                # print(obs['robot_0']['camera_pose'])
                image = Image.fromarray(img)
                image.show()
except KeyboardInterrupt:
    pass

env.close()
