from copy import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.prm import PRM
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG
from lab_ur_stack.manipulation.robot_with_motion_planning import to_canonical_config
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from scipy.spatial.transform import Rotation as R


@dataclass
class ItemPosition:
    name: str
    position: Sequence[float]
    pick_start_offset: Sequence[float]
    pickup_ee_rz: float = 0

@dataclass()
class ManipulatedObject:
    name: str
    grasp_width: float | None = None


item_positions = [
 ItemPosition("top_shelf_left", (-0.796, -0.189, 0.524), pick_start_offset=(0.1, 0, 0)),
 ItemPosition("top_shelf_right", (-0.796, 0.084, 0.524), pick_start_offset=(0.1, 0, 0)),
 ItemPosition("middle_shelf", (-0.781, -0.055, 0.287), pick_start_offset=(0.1, 0, 0.02)),
]

objects = [
    ManipulatedObject("red_cup"),
    ManipulatedObject("blue_cup"),
    ManipulatedObject("soda_can", grasp_width=57),
]
objects_dict = {obj.name: obj for obj in objects}


mp = POManMotionPlanner()
gt = GeometryAndTransforms.from_motion_planner(mp)

r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
r2_controller = ManipulationControllerVG(ur5e_2["ip"], ur5e_2["name"], mp, gt)
r1_controller.speed, r1_controller.acceleration = 0.5, .6
r2_controller.speed, r2_controller.acceleration = 0.5, .6

# r1_controller.plan_and_move_home()
r2_controller.plan_and_move_home()

item = item_positions[0]
pos = list(item.position)
pos[0] += 0.
width = objects_dict["soda_can"].grasp_width
r1_controller.pick_up_at_angle(pos, item.pick_start_offset, grasp_width=width)

item = item_positions[2]
r1_controller.put_down_at_angle(item.position, item.pick_start_offset)

# # test picking up:
# # item = ItemPosition("test_item", position=(0.35, -0.45, 0.6), pick_start_offset=(0, 0.05, 0.05), pickup_ee_rz=-np.pi)
# item = item_positions[0]
#
# # Compute approach vector (from offset to target)
# offset_position = np.array([
#     item.position[0] + item.pick_start_offset[0],
#     item.position[1] + item.pick_start_offset[1],
#     item.position[2] + item.pick_start_offset[2]
# ])
#
# # default_direciton = np.array([0, 0, 1])
# approach_vector = -np.array(item.pick_start_offset)
# approach_vector /= np.linalg.norm(approach_vector)
#
# mat_z = approach_vector
# mat_x = np.array([1, 0, 0])
# if np.linalg.norm(np.cross(mat_z, mat_x)) < 1e-6:
#     mat_x = np.array([0, 1, 0])
# mat_y = np.cross(mat_z, mat_x)
# rotation_matrix = np.array([mat_x, mat_y, mat_z]).T
# # add rotation around z axis by item.pickup_ee_rz
# rotation_matrix = rotation_matrix @ R.from_euler("z", item.pickup_ee_rz).as_matrix()
# axis_angle = R.from_matrix(rotation_matrix).as_rotvec()
#
# pick_up_start_pose = offset_position.tolist() + axis_angle.tolist()
#
# pick_up_start_config = r1_controller.find_ik_solution(pick_up_start_pose, for_down_movement=False)
# pick_up_start_config = to_canonical_config(pick_up_start_config)
# r1_controller.plan_and_moveJ(pick_up_start_config, speed=0.5, acceleration=0.6)
# item_pose = np.concatenate((item.position, axis_angle))
# r1_controller.moveL(item_pose, speed=0.1, acceleration=0.6)
# r1_controller.grasp()
# r1_controller.moveL(pick_up_start_pose, speed=0.1, acceleration=0.6)



# prm = PRM.load_roadmap(mp, "roadmap_ur5e_1.npy")

# path = prm.find_path(r1_controller.getActualQ(), red_cup_config)
#
# r1_controller.move_path(path, speed=0.3, acceleration=0.6)


pass