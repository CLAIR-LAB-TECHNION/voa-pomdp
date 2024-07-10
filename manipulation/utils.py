from utils.workspace_utils import (sample_block_positions, stack_position_r2frame,
                                   workspace_x_lims_default, workspace_y_lims_default)
from manipulation.manipulation_controller import ManipulationController


def ur5e_2_distribute_blocks_in_workspace(n_blocks,
                                          robot_controller: ManipulationController,
                                          ws_lim_x=workspace_x_lims_default,
                                          ws_lim_y=workspace_y_lims_default,
                                          stack_position_ur5e_2_frame=stack_position_r2frame,
                                          min_dist=0.06):
    """
    pickup n_blocks from stack_position and distribute them in the workspace with at least min_dist between them.
    """
    block_positions = sample_block_positions(n_blocks, ws_lim_x, ws_lim_y, min_dist)
    stack_position_world = robot_controller.gt.point_robot_to_world(robot_controller.robot_name,
                                                                    [*stack_position_ur5e_2_frame, 0.])

    start_height = 0.1 + 0.05 * n_blocks
    for block_pos in block_positions:
        start_height -= 0.05
        robot_controller.pick_up(stack_position_world[0], stack_position_world[1], rz=0, start_height=start_height)
        robot_controller.put_down(block_pos[0], block_pos[1], rz=0)

    return block_positions


def ur5e_2_collect_blocks_from_positions(block_positions, robot_controller: ManipulationController,
                                        stack_position_ur5e_2_frame=stack_position_r2frame):
    """
    collect blocks from block_positions and stack them at stack_position
    """
    stack_position_world = robot_controller.gt.point_robot_to_world(robot_controller.robot_name,
                                                                    [*stack_position_ur5e_2_frame, 0.])

    put_down_start_height = 0.1
    for block_pos in block_positions:
        put_down_start_height += 0.05
        robot_controller.pick_up(block_pos[0], block_pos[1], rz=0)
        robot_controller.put_down(stack_position_world[0], stack_position_world[1], rz=0,
                                  start_height=put_down_start_height)

