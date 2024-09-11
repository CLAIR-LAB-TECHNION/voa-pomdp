import numpy as np

from multi_mujoco.block_stacking_simulator.block_stacking_simulator import BlockStackingSimulator

blocks_pos = [[-0.8, -0.8, 0.04],
              [-0.8, -0.7, 0.04],
              [-0.65, -0.72, 0.04],
              [-0.6, -0.6, 0.04],]

simulator = BlockStackingSimulator(False)
simulator.reset(randomize=False, block_positions=blocks_pos)
simulator.pick_up(agent='ur5e_2', x=-0.65, y=-0.72)