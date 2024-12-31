import time
import typer
import numpy as np
from lab_ur_stack.manipulation.robot_with_motion_planning import to_canonical_config
from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_with_depth_as_image
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms


def sample_help_config(gt,
                       lookat_max_offset=0.4,
                       verangle_min=20,
                       verangle_max=90,
                       horangle_min=-10,
                       horangle_max=60,
                       distance_min=0.4,
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
def generate_configs(n_samples: int = 50,
                     save_to_file_name="help_configs.npy",
                     lookat_max_offset: float = 0.4,
                     verangle_min: int = 20,
                     verangle_max: int = 90,
                     horangle_min: int = -20,
                     horangle_max: int = 90,
                     distance_min: float = 0.35,
                     distance_max: float = 2.,
                     min_dist_between_poses: float = .5):
    gt = GeometryAndTransforms.build()

    help_configs = []
    poses = []
    while len(help_configs) < n_samples:
        help_config = sample_help_config(gt=gt,
                                         lookat_max_offset=lookat_max_offset,
                                         verangle_min=verangle_min,
                                         verangle_max=verangle_max,
                                         horangle_min=horangle_min,
                                         horangle_max=horangle_max,
                                         distance_min=distance_min,
                                         distance_max=distance_max, )
        help_config = to_canonical_config(help_config)
        pose = gt.robot_ee_to_world_transform("ur5e_1", help_config)
        pose = gt.se3_to_4x4(pose).flatten()

        # if this config is too close to an existing one, skip it
        if len(help_configs) > 0:
            # min_dist = np.min(np.linalg.norm(np.array(help_configs) - np.array(help_config), axis=1))
            # print(f"min dist to existing configs: {min_dist}")
            # if min_dist < 1.:
            #     print(f"skipping, too close to existing config, min dist: {min_dist}")
            #     continue

            min_dist_pose = np.min(np.linalg.norm(np.array(poses) - np.array(pose), axis=1))
            print(f"min dist to existing poses: {min_dist_pose}")
            if min_dist_pose < min_dist_between_poses:
                print(f"skipping, too close to existing pose, min dist: {min_dist_pose}")
                continue

        help_configs.append(help_config)
        poses.append(pose)
        print("saving...")

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


@app.command(
    context_settings={"ignore_unknown_options": True})
def detect_from_configs(file_name="help_configs.npy",):
    from lab_ur_stack.camera.realsense_camera import RealsenseCamera
    from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
    from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1
    from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
    from matplotlib import pyplot as plt

    import os, sys
    ci_build_and_not_headless = False
    try:
        from cv2.version import ci_build, headless
        ci_and_not_headless = ci_build and not headless
    except:
        pass
    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_FONTDIR")

    help_configs = np.load(file_name)
    gt = GeometryAndTransforms.build()
    gt.motion_planner.visualize()

    camera = RealsenseCamera()
    r1_controller = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_1["ip"], ur5e_1["name"])
    position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    for config in help_configs:
        r1_controller.moveJ(config)

        time.sleep(0.2)
        actual_config = r1_controller.getActualQ()
        rgb, depth = camera.get_frame_rgb()
        positions, annotations = position_estimator.get_block_positions_depth(rgb, depth, actual_config)
        plot_im = detections_plots_with_depth_as_image(annotations[0], annotations[1], annotations[2], positions,
                                                       workspace_x_lims_default, workspace_y_lims_default,)
        plt.imshow(plot_im)
        plt.show()


if __name__ == '__main__':
    app()
