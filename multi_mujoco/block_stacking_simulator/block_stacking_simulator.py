import numpy as np

from multi_mujoco.mujoco_env.voa_world import WorldVoA
from multi_mujoco.motion_planning.motion_executor import MotionExecutor

from multi_mujoco.mujoco_env.common.ur5e_fk import forward


# CONF = [1.0980085926014562,
#         -1.1826384468718751,
#         0.8487687291028511,
#         -1.3704098689960071,
#         -1.570138543721137,
#         -2.2891771777909833]


class BlockStackingSimulator:
    def __init__(self, visualize_mp=True):
        self.mujoco_env = WorldVoA()
        self.motion_executor = MotionExecutor(env=self.mujoco_env)
        self.current_state = None
        self.reset()
        self.tower_counter = 0
        self.tower_pos = [-0.45, -1.15]

        if visualize_mp:
            self.motion_executor.motion_planner.visualize()

    def reset(self, randomize=True, block_positions=None):
        """
        Reset the object positions in the simulation.
        Args:
            randomize: if True, randomize the positions of the blocks, otherwise set them to initial positions.
            block_positions: a list of block positions to set the blocks to. If provided, randomize will be ignored.
        """
        if block_positions:
            self.current_state = self.motion_executor.reset(randomize=False, block_positions=block_positions)
        else:
            self.current_state = self.motion_executor.reset(randomize=randomize)

    def close(self):
        self.mujoco_env.close()

    def sense_for_block(self, agent, x, y):
        return self.motion_executor.sense_for_block(agent, x, y)

    def pick_up(self, agent, x, y):
        return self.motion_executor.pick_up(agent, x, y)

    def stack(self, agent):
        self.motion_executor.put_down(agent, x=self.tower_pos[0], y=self.tower_pos[1])
        self.tower_counter += 1

    def sense_camera(self, camera_position, camera_orientation):
        # success, _, _ = self.motion_executor.move_to_pose(agent, camera_position, camera_orientation)
        # if success:
        return self.mujoco_env.render_image_from_pose(np.array(camera_position), np.array(camera_orientation))

