import json
import os
from typing import List

import cv2
import numpy as np
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation.manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2
from camera.realsense_camera import RealsenseCamera
import time
from PIL import Image


stack_position_r2frame = [-0.3614, 0.1927]

workspace_x_lims = [-0.9, -0.54 ]
workspace_y_lims = [-1.0, -0.55]

sensing_configs = [[-2.04893, -2.50817, -0.00758, -1.96019, 1.51035, 1.0796],
                    [-1.8152335325824183, -2.732894559899801, -0.5337811708450317, -1.1349691313556214, 1.0154942274093628, 1.1909868717193604],
                    [-2.29237, -2.50413, -0.76933, -1.37775, 1.53721, -0.74012],
                    [-1.54191, -1.8467, -2.39349, 0.27255, 0.80462, 0.63606],
                    [-1.51508, -2.15687, -1.61078, -0.84643, 0.68978, 1.52553],
                    [-0.13548, -1.67402, -1.56538, -2.86094, 1.10085, 1.63128],
                    [-1.4104, -1.74145, -0.18662, -2.55688, 1.09938, 1.80797]]


app = typer.Typer()


def valid_position(x, y, block_positions):
    if x is None or y is None:
        return False
    for block_pos in block_positions:
        if np.abs(block_pos[0] - x) < 0.05 or np.abs(block_pos[1]) - y < 0.05:
            return False
    return True


def sample_block_positions(n_blocks, workspace_x_lims, workspace_y_lims):
    ''' sample n_blocks positions within the workspace limits, spaced at least by 0.05m in each axis '''
    block_positions = []
    for i in range(n_blocks):
        x = None
        y = None
        while valid_position(x, y, block_positions) is False:
            x = np.random.uniform(*workspace_x_lims)
            y = np.random.uniform(*workspace_y_lims)
        block_positions.append([x, y])

    return block_positions


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 5,
         repeat: int = 4):
    camera = RealsenseCamera()

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed = 0.75
    r1_controller.acceleration = 0.75
    r2_controller.speed = 1.5
    r2_controller.acceleration = 1.5

    stack_position = gt.point_robot_to_world(ur5e_2["name"], (*stack_position_r2frame, 0.2))

    for i in range(repeat):
        r1_controller.move_home(speed=0.5, acceleration=0.5)

        # sample positions:
        block_positions = sample_block_positions(n_blocks, workspace_x_lims, workspace_y_lims)

        # distribute blocks
        for block_pos in block_positions:
            r2_controller.pick_up(stack_position[0], stack_position[1], rz=0, start_height=0.25)
            r2_controller.put_down(block_pos[0], block_pos[1], rz=0)

        # clear out for images
        r2_controller.plan_and_move_to_xyzrz(stack_position[0], stack_position[1], 0.2, 0)

        # create dir for this run
        datetime = time.strftime("%Y%m%d-%H%M%S")
        run_dir = f"collected_images/{n_blocks}blocks_{datetime}/"
        os.makedirs(run_dir, exist_ok=True)

        # Save block positions
        metadata = {
            "ur5e_1_config": [],
            "ur5e_2_config": [],
            "block_positions": block_positions,
            "images_depth": [],
            "images_rgb": [],
        }

        for idx, c in enumerate(sensing_configs):
            if motion_planner.is_config_feasible("ur5e_1", c) is False:
                print(f"Config {c} is not feasible, probably collides with other robot")
                continue
            r1_controller.plan_and_moveJ(c)
            im, depth = camera.get_frame()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plotable_depth = camera.plotable_depth(depth)

            image_filename_png, image_filename_npy = f"image_{idx}.png", f"image_{idx}.npy"
            depth_filename_png, depth_filename_npy = f"plotable_depth_{idx}.png", f"depth_{idx}.npy"

            Image.fromarray(im).save(os.path.join(run_dir, image_filename_png))
            np.save(os.path.join(run_dir, image_filename_npy), im)
            Image.fromarray(plotable_depth).save(os.path.join(run_dir, depth_filename_png))
            np.save(os.path.join(run_dir, depth_filename_npy), depth)

            # Save metadata
            metadata["ur5e_1_config"].append(c)
            metadata["ur5e_2_config"].append(r2_controller.getActualQ())
            metadata["images_rgb"].append(image_filename_npy)
            metadata["images_depth"].append(depth_filename_npy)

        # Save metadata to JSON
        metadata_path = os.path.join(run_dir, "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        r1_controller.move_home(speed=0.5, acceleration=0.5)

        # recollect blocks:
        for block_pos in block_positions:
            r2_controller.pick_up(block_pos[0], block_pos[1], rz=0)
            r2_controller.put_down(stack_position[0], stack_position[1], rz=0,  start_height=0.25)


if __name__ == "__main__":
    app()

