import numpy as np

from multi_mujoco.mujoco_env.voa_world import WorldVoA

if __name__ == '__main__':
    world = WorldVoA()
    world.move_to('robot_0', np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    world.close()
