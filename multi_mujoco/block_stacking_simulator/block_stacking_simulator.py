from multi_mujoco.motion_planning.simulation_motion_planner import SimulationMotionPlanner
from multi_mujoco.mujoco_env import mujoco_env
from multi_mujoco.mujoco_env.tasks import NullTask


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
    def __init__(self,):
        self.mujoco_env = mujoco_env.MujocoEnv.from_cfg(cfg=muj_env_config, render_mode="human", frame_skip=5)
        self.motion_planner = SimulationMotionPlanner()

        obs, info = self.mujoco_env.reset()
        self.mj_model = info["privileged"]["model"]
        self.mj_data = info["privileged"]["data"]

        pass

    def reset(self, block_positions):
        """
        TODO: set block positions given the list of block posions, implement set_all_block_positions
        TODO: for that. Then implement get block positions.
        TODO then move all to blocks_manager class.
        """
        # set blocks positions
        # reset mujoco env.

    def set_block_position(self, block_id, position):
        joint_name = f"block{block_id+1}_fj"
        joint_id = self.mj_model.joint(joint_name).id
        pos_adrr = self.mj_model.jnt_qposadr[joint_id]
        self.mj_data.qpos[pos_adrr:pos_adrr+3] = position


    def set_all_block_positions(self, positions):
        pass

    def get_block_position(self, block_id):
        pass

    def get_all_block_positions(self):
        pass



if __name__ == '__main__':
    simulator = BlockStackingSimulator()
    pass
