import time
import typer
import numpy as np
from lab_ur_stack.manipulation.utils import to_canonical_config
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms


def sample_help_config(gt,
                       lookat_max_offset=0.2,
                       verangle_min=30,
                       verangle_max=90,
                       horangle_min=-10,
                       horangle_max=60,
                       distance_min=0.5,
                       distance_max=1.5,
                       ws_x_lims=workspace_x_lims_default,
                       ws_y_lims=workspace_y_lims_default,
                       ):
    """
    sample config for robot1 to take image from to help.
    The config is sampled according to the rules described in the implementation
    @return: Configuration for robot1, with canonical joint angels (between -pi and pi)
    """
    help_config = None

    # sample camera poses until one that is reachable by the robot is found:
    while help_config is None:
        # first sample point to look at (center of image axis),
        # it will be uniformly distributed around the workspace center
        ws_center_x = np.mean(ws_x_lims)
        ws_center_y = np.mean(ws_y_lims)
        lookat_x = np.random.uniform(ws_center_x - lookat_max_offset, ws_center_x + lookat_max_offset)
        lookat_y = np.random.uniform(ws_center_y - lookat_max_offset, ws_center_y + lookat_max_offset)
        lookat = [lookat_x, lookat_y, 0]

        # sample horizontal and vertical angles to look from,
        # uniform around the intervals of angles that maybe reachable
        verangle = np.random.uniform(verangle_min, verangle_max)
        horangle = np.random.uniform(horangle_min, horangle_max)

        # sample distance from lookat, below 0.5 depth camera doesn't work
        distance = np.random.uniform(distance_min, distance_max)

        print(f"trying for lookat: {lookat}, verangle: {verangle}, horangle: {horangle}, distance: {distance}")

        # will return None if no valid config is found in these settings, it will look at different
        # rotations angles around camera axis
        help_config = lookat_verangle_horangle_distance_to_robot_config(lookat, verangle, horangle, distance,
                                                                        gt, "ur5e_1")

    help_config = to_canonical_config(help_config)
    print(f"found help config: {help_config}")
    return help_config


app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True})
def generate_configs(n_samples: int = 1000,
                     save_to_file_name="help_configs.npy",
                     lookat_max_offset: float = 0.2,
                     verangle_min: int = 30,
                     verangle_max: int = 90,
                     horangle_min: int = -10,
                     horangle_max: int = 60,
                     distance_min: float = 0.5,
                     distance_max: float = 1.5):
    gt = GeometryAndTransforms.build()

    help_configs = []
    for i in range(n_samples):
        help_config = sample_help_config(gt=gt,
                                         lookat_max_offset=lookat_max_offset,
                                         verangle_min=verangle_min,
                                         verangle_max=verangle_max,
                                         horangle_min=horangle_min,
                                         horangle_max=horangle_max,
                                         distance_min=distance_min,
                                         distance_max=distance_max, )
        help_configs.append(help_config)

    help_configs = np.array(help_configs)
    np.save(save_to_file_name, help_configs)


@app.command(
    context_settings={"ignore_unknown_options": True})
def visualize_configs(file_name="help_configs.npy",
                      time_per_batch: float = 2,
                      batch_size: int = 5):
    help_configs = np.load(file_name)
    gt = GeometryAndTransforms.build()
    gt.motion_planner.visualize()

    colors = [(0.5, 0.5, 0, 0.7), (0.5, 0, 0.5, 0.7), (0, 0.5, 0.5, 0.7), (1., 0.5, 0.5, 0.7),
              (0, 0, 0.5, 0.7), (0.5, 1., 0.3, 0.7)]

    for i in range(0, len(help_configs), batch_size):
        batch = help_configs[i:i + batch_size]
        for j, config in enumerate(batch):
            color = colors[j%len(colors)]
            gt.motion_planner.vis_config("ur5e_1", config, str(j), color)
        time.sleep(time_per_batch)


if __name__ == '__main__':
    app()
