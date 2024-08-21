import numpy as np

from multi_mujoco.mujoco_env.voa_world import WorldVoA
from multi_mujoco.motion_planning.motion_executor import MotionExecutor


class BlockStackingSimulator:
    def __init__(self):
        self.mujoco_env = WorldVoA()
        self.motion_executor = MotionExecutor(env=self.mujoco_env)
        self.current_state = None
        self.reset()
        self.tower_counter = 0
        self.tower_pos = [-0.3, -1.]

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

    def pick_up(self, agent, x, y):
        self.motion_executor.pick_up(agent, x, y)

    def stack(self, agent):
        z = 0.04 + 0.05*self.tower_counter
        self.motion_executor.put_down(agent, x=self.tower_pos[0], y=self.tower_pos[1], z=z)
        self.tower_counter += 1

    def sense_camera(self, agent, camera_position, camera_orientation):
        success, _, obs = self.motion_executor.move_to_pose(agent, camera_position, camera_orientation)
        return obs['robots_camera'][agent] if success else None  # A list of two, first is image, second is camera pose


if __name__ == '__main__':
    simulator = BlockStackingSimulator()

    simulator.motion_executor.move_to_config('ur5e_2', np.array([0.0, -1.2, 0.8419, -1.3752, -1.5739, -2.3080]))
    box = simulator.mujoco_env.get_state()['object_positions']['block0_fj']
    simulator.pick_up('ur5e_2', box[0], box[1])
    simulator.stack('ur5e_2')

    box = simulator.mujoco_env.get_state()['object_positions']['block1_fj']
    simulator.pick_up('ur5e_2', box[0], box[1])
    simulator.stack('ur5e_2')

    simulator.close()
