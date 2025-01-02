from copy import copy
import numpy as np
import typer
from lab_po_manipulation.env_configurations.objects_and_positions import positions_dict, objects_dict,\
    save_grasps_to_json, load_grasps_from_json
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.prm import default_joint_limits_low, default_joint_limits_high
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1
import json
from pathlib import Path


app = typer.Typer()


def generate_grasp_configs(
    position_name:str = typer.Argument(..., help="Name of the position to generate grasp configurations for")
):
    mp = POManMotionPlanner()


def test_execution_grasp_config(position_name, offset, ee_rz,
                                robot_controller: ManipulationController2FG):
    position = positions_dict[position_name]
    pos = list(position.position)
    robot_controller.pick_up_at_angle(pos, offset, ee_rz)
    robot_controller.put_down_at_angle(pos, offset, ee_rz)


if __name__ == "__main__":
    mp = POManMotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(mp)
    r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, .75

    grasp_configs = [
        ([0.1, 0, 0.01], 0),
        ([0.1, 0, 0.035], 0),
        ([0.08, 0.025, 0.03], 0),
    ]

    # for offset, ee_rz in grasp_configs:
    #     test_execution_grasp_config("middle_shelf", offset, ee_rz, r1_controller)

    save_grasps_to_json("middle_shelf", grasp_configs)
