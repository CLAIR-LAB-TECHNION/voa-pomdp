import numpy as np
from matplotlib import pyplot as plt
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack

blocks_pos = [[-0.8, -0.8],
              [-0.8, -0.7],
              [-0.65, -0.72],
              [-0.6, -0.6],]


simulator = BlockStackingSimulator(visualize_mp=False, max_steps=7, render_sleep_to_maintain_fps=True,
                                   render_mode="human")

help_configs = np.load("../experiments_lab/configurations/help_configs.npy")

for i in range(10):
    blocks_pos[0][0] += 0.01
    print("resetting with blocks_pos", blocks_pos)
    simulator.reset(block_positions=blocks_pos)


    im = simulator.sense_camera_r1([-0.5567808869537451, -1.215127556940363, -1.8444974285294637, 1.258726315197987, -0.6591467161558847, -0.20351355710407049])
    plt.imshow(im)
    plt.show()

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
