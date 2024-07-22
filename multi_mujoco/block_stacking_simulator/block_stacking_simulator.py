from multi_mujoco.motion_planning.simulation_motion_planner import SimulationMotionPlanner
from multi_mujoco.mujoco_env import mujoco_env
from multi_mujoco.mujoco_env.tasks import NullTask
from object_manager import ObjectManager

muj_env_config = dict(
    scene=dict(
        resource='clairlab',
        render_camera='top-right'
    ),
    robots=dict(
        robot_0=dict(
            resource='ur5e',
            # attachments=['adhesive_gripper'],
            base_pos=[0, 0, 0],  # Example base position
            base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
            privileged_info=True,
        ),
        robot_1=dict(
            resource='ur5e',
            # attachments=['adhesive_gripper'],
            base_pos=[-0.76, -1.33, 0],  # Example base position
            base_orient=[0, 0, 0, 1],  # Example base orientation (quaternion)
            privileged_info=True,
        ),
    ),
    tasks=dict(
        robot_0=NullTask,
        robot_1=NullTask,
    ),
)


class BlockStackingSimulator:
    def __init__(self, ):
        self.mujoco_env = mujoco_env.MujocoEnv.from_cfg(cfg=muj_env_config, render_mode="human", frame_skip=5)
        self.motion_planner = SimulationMotionPlanner()

        obs, info = self.mujoco_env.reset()
        self.mj_model = info["privileged"]["model"]
        self.mj_data = info["privileged"]["data"]
        self.object_manager = ObjectManager(self.mj_model, self.mj_data)

    def reset(self, randomize=False, block_positions=None):
        """
        Reset the object positions in the simulation.
        Args:
            randomize: if True, randomize the positions of the blocks, otherwise set them to initial positions.
            block_positions: a list of block positions to set the blocks to. If provided, randomize will be ignored.
        """
        if block_positions:
            self.object_manager.reset(randomize=False, block_positions=block_positions)
        else:
            self.object_manager.reset(randomize=randomize)


if __name__ == '__main__':
    simulator = BlockStackingSimulator()
    pass
