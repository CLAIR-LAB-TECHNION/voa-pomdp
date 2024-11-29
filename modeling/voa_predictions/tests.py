import numpy as np
from matplotlib import pyplot as plt
import typer
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from experiments_sim.utils import build_position_estimator
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from modeling.voa_predictions.utils import in_r1_fov


app = typer.Typer()

@app.command()
def test_in_fov_sim():
    tests_help_config_indexes = [0, 0, 0, 5, 5]
    block_positions = [
        [[-0.7, -0.7], [2, 2], [2, 2], [2, 2]],
        [[-0.7, -0.7], [-0.9, -0.9], [2, 2], [2, 2]],
        [[-0.7, -0.7], [-0.9, -0.9], [-1.1, -1.1], [2, 2]],
        [[-0.7, -0.71], [-0.9, -0.9], [-1.1, -1.1], [2, 2]],
        [[-0.7, -0.69], [-0.9, -0.9], [-1.1, -1.1], [2, 2]],
    ]

    block_positions_3d = []
    for curr_pos_list in block_positions:
        block_positions_3d.append([pos + [0.025] for pos in curr_pos_list])

    # create env and render from some help configs
    # plot the image and write how many points are in the fov at title
    env = BlockStackingSimulator(visualize_mp=False, render_mode='None',)
    gt = GeometryAndTransforms(env.motion_executor.motion_planner,
                               cam_in_ee=-np.array(env.helper_camera_translation_from_ee))
    sim_help_cofnigs = np.load("./experiments_sim/configurations/sim_help_configs_40.npy")

    for block_pos, block_pos_3d, help_config in zip(block_positions, block_positions_3d, sim_help_cofnigs[tests_help_config_indexes]):
        env.reset(block_positions=block_pos)
        im, actual_config = env.sense_camera_r1(help_config)
        print(sim_help_cofnigs)
        print(help_config)
        in_fov = in_r1_fov(np.array(block_pos_3d), actual_config, gt,
                           env.mujoco_env.get_robot_cam_intrinsic_matrix())

        plt.imshow(im)
        plt.title(f"{sum(in_fov)} points in fov")
        plt.show()


@app.command()
def d():
    pass


if __name__ == "__main__":
    app()
