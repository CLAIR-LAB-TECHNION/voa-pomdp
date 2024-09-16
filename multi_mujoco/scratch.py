import numpy as np

from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack

blocks_pos = [[-0.8, -0.8],
              [-0.8, -0.7],
              [-0.65, -0.72],
              [-0.6, -0.6],]


simulator = BlockStackingSimulator(visualize_mp=False, max_steps=7, render_sleep_to_maintain_fps=False)

for i in range(10):
    blocks_pos[0][0] += 0.01
    print("resetting with blocks_pos", blocks_pos)
    simulator.reset(block_positions=blocks_pos)
    simulator.motion_executor.plan_and_moveJ("ur5e_1", [+1.0776e+00, -1.57137098e+00, 1.06499581e-04, -1.57132691e+00,
                                                        -2.73702160e-06, 4.52766e-07])
    obs = None

    while True:
        # sample action:
        i = np.random.randint(0, 2)
        x, y = np.random.uniform(-0.9, -0.54), np.random.uniform(-1.0, -0.55)
        action = ActionSense(x, y) if i == 0 else ActionAttemptStack(x, y)
        obs, reward = simulator.step(action)
        print("-----------------")
        print(action)
        print(obs)
        print("reward", reward)

        if obs.steps_left == 0:
            break


# for bpos in blocks_pos:
#     simulator.pick_up(agent='ur5e_2', x=bpos[0], y=bpos[1])
#     simulator.stack(agent='ur5e_2')
