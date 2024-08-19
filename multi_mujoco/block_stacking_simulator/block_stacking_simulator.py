import numpy as np

from multi_mujoco.mujoco_env.voa_world import WorldVoA
from multi_mujoco.motion_planning.motion_executor import MotionExecutor


class BlockStackingSimulator:
    def __init__(self):
        self.mujoco_env = WorldVoA()
        self.motion_executor = MotionExecutor(env=self.mujoco_env)
        self.current_state = None
        self.reset()

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

    def sense_height(self, agent, x, y):
        self.motion_executor.move_and_detect_height(agent, x, y)

    def sense_camera(self, agent, camera_position, camera_orientation):
        success, _, obs = self.motion_executor.move_to_pose(agent, camera_position, camera_orientation)
        return obs['robots_camera'][agent] if success else None  # A list of two, first is image, second is camera pose


if __name__ == '__main__':
    simulator = BlockStackingSimulator()
    FACING_DOWN_R = [[0, 0, -1],
                     [0, 1, 0],
                     [1, 0, 0]]
    simulator.motion_executor.move_to_config('ur5e_2', np.array([1., -1., -1.5, -1.5, 0., 0.]))
    # simulator.sense_camera('ur5e_1', [0.2, 0.3, 0.2], FACING_DOWN_R)
    simulator.sense_height('ur5e_2', -0.7, -0.7)
    simulator.close()
