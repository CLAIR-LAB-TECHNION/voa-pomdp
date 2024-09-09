import os
import time
import logging
from dataclasses import dataclass

import pandas as pd
import typer
import numpy as np
import chime
from matplotlib import pyplot as plt

from experiments_lab.block_stacking_env import LabBlockStackingEnv
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCameraWithRecording
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.pouct_planner_policy import POUCTPolicy
from experiments_lab.experiment_manager import ExperimentManager


table_corner = [-0.3986, -1.5227]

camera_robot_config = [-0.9148362318622034, -2.383674760858053, -0.2041454315185547, 3.377373381251953, 0.7095475196838379, 2.552877187728882]


@dataclass
class ObjectData:
    name: str
    pickup_offset_from_corner: list
    target_position: list
    target_position_2: list
    start_height: float
    rz_init: float
    rz_final: float
    grasp_force_scale: float

def place_object(object_data: ObjectData, controller: ManipulationController):
    """ pick up the object from the table corner and place it at the target position """
    pickup_position = np.array(table_corner) + np.array(object_data.pickup_offset_from_corner)
    controller.plan_and_move_to_xyzrz(-0.3, -1., 0.45, 0)
    controller.pick_up(*pickup_position,
                       object_data.rz_init,
                       start_height=object_data.start_height,
                       force_scale=object_data.grasp_force_scale)
    if object_data.name == "spray_bottle":
        speed, acceleration = 0.8, 1.
    else:
        speed, acceleration = controller.speed, controller.acceleration
    controller.plan_and_move_to_xyzrz(-0.3, -1., 0.45, 0, speed=speed, acceleration=acceleration)
    controller.put_down(*object_data.target_position,
                        object_data.rz_final,
                        start_height=object_data.start_height,
                        speed=speed,
                        acceleration=acceleration)

def pick_and_place_object(object_data: ObjectData, controller: ManipulationController):
    """ pick and place from target1 to target2"""
    controller.pick_up(*object_data.target_position,
                       object_data.rz_final,
                       start_height=object_data.start_height,
                       force_scale=object_data.grasp_force_scale)
    if object_data.name == "spray_bottle":
        speed, acceleration = 0.8, 1.
    else:
        speed, acceleration = controller.speed, controller.acceleration
    controller.put_down(*object_data.target_position_2,
                        object_data.rz_final,
                        start_height=object_data.start_height,
                        speed=speed,
                        acceleration=acceleration)



objects = [
    ObjectData("spray_bottle",
               pickup_offset_from_corner=[-0.03, 0],
               target_position=[-0.77, -0.6],
               target_position_2=[-0.93, -0.9],
               start_height=0.35,
               rz_init=np.pi / 2,
               rz_final=0,
               grasp_force_scale=1.),
    ObjectData("bottle",
                      pickup_offset_from_corner=[0,0],
                      target_position=[-0.7, -0.92],
                      target_position_2=[-0.77, -0.6],
                      start_height=0.33,
                      rz_init=0,
                      rz_final=0,
                      grasp_force_scale=0.7),
           ObjectData("cola_can",
                      pickup_offset_from_corner=[0, 0],
                      target_position=[-0.6, -0.72],
                      target_position_2=[-0.6, -0.89],
                      start_height=0.2,
                      rz_init=0,
                      rz_final=0,
                      grasp_force_scale=0.2),
           ]

app = typer.Typer()
@app.command(context_settings={"ignore_unknown_options": True})
def main():
    camera = RealsenseCameraWithRecording()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.4, 0.75
    r2_controller.speed, r2_controller.acceleration = 2., 3.

    r1_controller.moveJ(camera_robot_config)

    # place objects
    # for obj in objects:
    #     print(f"ready to place {obj.name}. press enter to start")
    #     input()
    #     place_object(obj, r2_controller)
    #
    print("ready to pick and place objects. press enter to start")
    input()
    camera.start_recording("./objects_vid", fps=20)
    for obj in objects:
        time.sleep(2)
        pick_and_place_object(obj, r2_controller)
    camera.stop_recording()


if __name__ == "__main__":
    app()
