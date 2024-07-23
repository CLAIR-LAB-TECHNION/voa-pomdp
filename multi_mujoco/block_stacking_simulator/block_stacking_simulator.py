import numpy as np

from multi_mujoco.mujoco_env.voa_world import WorldVoA
from multi_mujoco.motion_planning.motion_executor import MotionExecutor


class BlockStackingSimulator:
    def __init__(self):
        self.mujoco_env = WorldVoA()
        self.motion_executor = MotionExecutor(env=self.mujoco_env)
        self.reset()

    def reset(self, randomize=True, block_positions=None):
        """
        Reset the object positions in the simulation.
        Args:
            randomize: if True, randomize the positions of the blocks, otherwise set them to initial positions.
            block_positions: a list of block positions to set the blocks to. If provided, randomize will be ignored.
        """
        if block_positions:
            self.motion_executor.reset(randomize=False, block_positions=block_positions)
        else:
            self.motion_executor.reset(randomize=randomize)

    def close(self):
        self.mujoco_env.close()


if __name__ == '__main__':
    simulator = BlockStackingSimulator()
    simulator.motion_executor.move_to_config('ur5e_1', np.array([2., -1.5, -1.5, -1.5, 0., 0.]))
    simulator.close()
    pass
