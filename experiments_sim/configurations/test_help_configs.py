""" make sure that all help configs are plannable and moveable to from home config"""

import numpy as np
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
import matplotlib.pyplot as plt
import math
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default


max_planning_time = 5
max_planning_time_retry = 10


simulator = BlockStackingSimulator(visualize_mp=True, render_sleep_to_maintain_fps=False,)
help_configs = np.load("sim_help_configs_40.npy")
# 4 corners:
block_pos = [[workspace_x_lims_default[0], workspace_y_lims_default[0]],
                [workspace_x_lims_default[0], workspace_y_lims_default[1]],
                [workspace_x_lims_default[1], workspace_y_lims_default[0]],
                [workspace_x_lims_default[1], workspace_y_lims_default[1]]]

images = []
for i, help_config in enumerate(help_configs):
    print(f"checking help_config {i}")
    simulator.reset(block_positions=block_pos)
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

    if success:
        im = simulator.sense_camera_r1(help_config)[0]
        images.append(im)

# plot all images in a grid for visual inspection

def plot_image_grid(images, figsize=(30, 20), dpi=300):
    # Calculate grid dimensions
    n_images = len(images)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)

    # Create figure with high resolution
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Plot each image
    for idx, img in enumerate(images):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


plot_image_grid(images)
# save plotted im
plt.savefig("help_configs_plotted.png")







