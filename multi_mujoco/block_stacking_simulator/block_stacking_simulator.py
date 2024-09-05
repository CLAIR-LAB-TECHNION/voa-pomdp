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
        self.tower_pos = [-0.3, -1.]

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

    def sense_height(self, agent, x, y):
        self.motion_executor.move_and_detect_height(agent, x, y)

    def pick_up(self, agent, x, y):
        self.motion_executor.pick_up(agent, x, y)

    def stack(self, agent):
        z = 0.04 + 0.05 * self.tower_counter
        self.motion_executor.put_down(agent, x=self.tower_pos[0], y=self.tower_pos[1], z=z)
        self.tower_counter += 1

    def sense_camera(self, camera_position, camera_orientation):
        # success, _, _ = self.motion_executor.move_to_pose(agent, camera_position, camera_orientation)
        # if success:
        return self.mujoco_env.render_image_from_pose(np.array(camera_position), np.array(camera_orientation))


if __name__ == '__main__':
    simulator = BlockStackingSimulator()

    FACING_DOWN_R = [[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]]

    simulator.motion_executor.move_to_config('ur5e_2', np.array([1.1, -1.2, 0.8419, -1.3752, -1.5739, -2.3080]))
    box = simulator.mujoco_env.get_state()['object_positions']['block0_fj']
    simulator.pick_up('ur5e_2', box[0], box[1])
    simulator.stack('ur5e_2')

    box = simulator.mujoco_env.get_state()['object_positions']['block1_fj']
    simulator.pick_up('ur5e_2', box[0], box[1])
    simulator.stack('ur5e_2')
    img = simulator.sense_camera([-0.17428016, -0.63361125, 0.59896269],
                                 [[0.24620755, -0.96735704, -0.06001828], [-0.9639793, -0.23797787, -0.11878736],
                                  [0.10062677, 0.08710273, -0.99110412]])

    simulator.close()
