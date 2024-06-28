from typing import List

import numpy as np
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation.manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2


stack_position_r2frame = [-0.3390, 0.1762]

workspace_x_lims = [-0.9, -0.54 ]
workspace_y_lims = [-1.0, -0.5]

app = typer.Typer()


def valid_position(x, y, block_positions):
    if x is None or y is None:
        return False
    for block_pos in block_positions:
        if np.abs(block_pos[0] - x) < 0.05 and np.abs(block_pos[1]) - y < 0.05:
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
def main(n_blocks: int = 3,):

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    # r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)

    stack_position = gt.point_robot_to_world(ur5e_2["name"], (*stack_position_r2frame, 0.2))

    # sample positions:
    block_positions = sample_block_positions(n_blocks, workspace_x_lims, workspace_y_lims)

    # distribute blocks
    for block_pos in block_positions:
        r2_controller.pick_up(stack_position[0], stack_position[1], rz=0)
        r2_controller.put_down(block_pos[0], block_pos[1], rz=0)

    # recollect blocks:
    for block_pos in block_positions:
        r2_controller.pick_up(block_pos[0], block_pos[1], rz=0)
        r2_controller.put_down(stack_position[0], stack_position[1], rz=0)


if __name__ == "__main__":
    app()

