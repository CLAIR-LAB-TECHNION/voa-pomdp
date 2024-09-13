import numpy as np

from multi_mujoco.block_stacking_simulator.block_stacking_simulator import BlockStackingSimulator

blocks_pos = [[-0.8, -0.8, 0.04],
              [-0.8, -0.7, 0.04],
              [-0.65, -0.72, 0.04],
              [-0.6, -0.6, 0.04],]


simulator = BlockStackingSimulator(False)
simulator.reset(randomize=False, block_positions=blocks_pos)

# for bpos in blocks_pos:
#     print(simulator.sense_for_block(agent='ur5e_2', x=bpos[0], y=bpos[1]))
#
# print(simulator.sense_for_block(agent='ur5e_2', x=-0.8, y=-.75))
# print(simulator.sense_for_block(agent='ur5e_2', x=-0.65, y=-.73))

for bpos in blocks_pos:
    simulator.pick_up(agent='ur5e_2', x=bpos[0], y=bpos[1])
    simulator.stack(agent='ur5e_2')
