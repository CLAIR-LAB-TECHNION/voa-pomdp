""" make sure that all help configs are plannable and moveable to from home config"""

import numpy as np
from experiments_sim.block_stacking_simulator import BlockStackingSimulator


max_planning_time = 5
max_planning_time_retry = 10


simulator = BlockStackingSimulator(visualize_mp=True, render_sleep_to_maintain_fps=False,)
help_configs = np.load("sim_help_configs_50.npy")

for i, help_config in enumerate(help_configs):
    print(f"checking help_config {i}")
    simulator.reset(block_positions=[[-1, -1], [-1, -0.9], [-0.9, -1], [-0.9, -0.9]])
    success = simulator.motion_executor.plan_and_moveJ("ur5e_1", help_config, speed=3., acceleration=3.)
    if not success:
        print(f"help_config {i} was not successful, retrying with more time"
              f" ({max_planning_time} > {max_planning_time_retry} seconds)")
        success = simulator.motion_executor.plan_and_moveJ("ur5e_1", help_config, speed=3., acceleration=3.,
                                                           max_planning_time=max_planning_time_retry)
        if not success:
            print(f"help_config {i} was not successful even with more time")
            print("help config: ", help_config)
        else:
            print(f"help_config {i} was successful with more time")
