from typing import List
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation.manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2
from klampt.math import se3
from klampt import vis
import numpy as np
import time


app = typer.Typer()


def lookat_verangle_distance_to_camera_transform(lookat, vertical_angle, distance, y_offset=0.3):
    """
    returns the camera se3 transform given the lookat point, vertical angle and distance.
    the camera will be in the same x as the lookat point, and y will be lookat[1] + y_offset
    :param lookat:
    :param vertical_angle:
    :param distance:
    :return:
    """
    vertical_angle = np.deg2rad(vertical_angle)

    # Calculate the camera position in the world frame
    delta_x = distance * np.cos(vertical_angle)
    delta_z = distance * np.sin(vertical_angle)
    camera_position = np.array([lookat[0] + delta_x, lookat[1] + y_offset, lookat[2] + delta_z])

    # Calculate the direction vector from the camera to the look-at point
    direction = lookat - camera_position
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # build rotation matrix from up, right, forward vectors
    # Assume the up vector is [1, 0, 0] for simplicity this is good because it's toward the workspace,
    # the camera will be aligned
    up = np.array([1, 0, 0])

    # Calculate the right vector
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)  # Normalize the right vector

    # Recalculate the up vector to ensure orthogonality
    up = np.cross(direction, right)

    # Create the rotation matrix, this is the world to camera rotation matrix
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 0] = right
    rotation_matrix[:, 1] = up
    rotation_matrix[:, 2] = direction
    # Invert the rotation matrix to get the camera to world rotation matrix
    rotation_matrix = np.linalg.inv(rotation_matrix)

    return rotation_matrix.flatten(), camera_position


def visualize_robot(motion_planner, gt, lookat, vertical_angle, distance, y_offset=0.3,):
    cam_transform = lookat_verangle_distance_to_camera_transform(lookat, vertical_angle, distance, y_offset)
    vis.add("cam", cam_transform)

    ee_transform = se3.mul(gt.camera_to_ee_transform(), cam_transform)
    config = motion_planner.ik_solve("ur5e_1", ee_transform)

    if config is not None:
        motion_planner.vis_config("ur5e_1", config)
    else:
        print("No solution found")


@app.command(
    context_settings={"ignore_unknown_options": True})
def main():
    workspace_limits_x = [-0.9, -0.54]
    workspace_limits_y = [-1.0, -0.55]
    workspace_center = [(workspace_limits_x[0] + workspace_limits_x[1]) / 2,
                        (workspace_limits_y[0] + workspace_limits_y[1]) / 2,
                        0]
    workspace_corners = [[workspace_limits_x[0], workspace_limits_y[0], 0.02],
                         [workspace_limits_x[1], workspace_limits_y[0], 0.02],
                         [workspace_limits_x[1], workspace_limits_y[1], 0.02],
                         [workspace_limits_x[0], workspace_limits_y[1], 0.02]]

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    motion_planner.visualize(beckend="PyQt5")
    for i, corner in enumerate(workspace_corners):
        motion_planner.show_point_vis(corner, f"c{i}")

    cam_transform = lookat_verangle_distance_to_camera_transform(workspace_center, 45, 0.5)
    vis.add("cam", cam_transform)
    ee_transform = se3.mul(gt.camera_to_ee_transform(), cam_transform)

    config = motion_planner.ik_solve("ur5e_1", ee_transform)

    motion_planner.vis_config("ur5e_1", config)
    # TODO what about collisions!
    # TODO assert that
    time.sleep(60)


if __name__ == "__main__":
    app()

