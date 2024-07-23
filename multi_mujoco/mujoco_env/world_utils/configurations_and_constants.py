import numpy as np

from multi_mujoco.mujoco_env.tasks.null_task import NullTask

# env_cfg = dict(
#     scene=dict(
#         resource='clairlab',
#         render_camera='top-right'
#     ),
#     robots=dict(
#         robot_0=dict(
#             resource='ur5e',
#             # mount='rethink_stationary',
#             attachments=['adhesive_gripper'],
#             base_pos=[0.1, -0.55, 0],  # Example base position
#             base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
#         ),
#         robot_1=dict(
#             resource='ur5e',
#             # mount='rethink_stationary',
#             attachments=['adhesive_gripper'],
#             base_pos=[-0.76, -1.33, 0],  # Example base position
#             base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
#         ),
#     ),
#     tasks=dict(
#         robot_0=NullTask,
#         robot_1=NullTask,
#     ),
# )

muj_env_config = dict(
    scene=dict(
        resource='clairlab',
        render_camera='top-right'
    ),
    robots=dict(
        ur5e_1=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[0, 0, 0],  # Example base position
            base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
            privileged_info=True,
        ),
        ur5e_2=dict(
            resource='ur5e',
            attachments=['adhesive_gripper'],
            base_pos=[-0.76, -1.33, 0],  # Example base position
            base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
            privileged_info=True,
        ),
    ),
    tasks=dict(
        ur5e_1=NullTask,
        ur5e_2=NullTask,
    ),
)

INIT_MAX_VELOCITY = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

# relative position of grasped object from end effector
grasp_offset = 0.02

frame_skip = 5
