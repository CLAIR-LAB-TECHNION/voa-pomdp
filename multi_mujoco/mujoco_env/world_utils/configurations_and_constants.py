import numpy as np

from multi_mujoco.mujoco_env.tasks.null_task import NullTask


muj_env_config = dict(
    scene=dict(
        resource='clairlab',
        render_camera='top-right'
    ),
    robots=dict(
        ur5e_1=dict(
            resource='ur5e',
            attachments=['robotiq_2f85'],
            base_pos=[0, 0, 0.01],
            base_orient=[0, 0, 0, 1],
            privileged_info=True,
        ),
        ur5e_2=dict(
            resource='ur5e',
            attachments=['robotiq_2f85'],
            base_pos=[-0.76, -1.33, 0.01],
            base_orient=[0, 0, 0, 1],
            privileged_info=True,
        ),
    ),
    tasks=dict(
        ur5e_1=NullTask,
        ur5e_2=NullTask,
    ),
)

INIT_MAX_VELOCITY = np.array([3]*6)

# relative position of grasped object from end effector
grasp_offset = 0.14

frame_skip = 5
