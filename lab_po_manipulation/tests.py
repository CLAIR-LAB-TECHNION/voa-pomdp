from dataclasses import dataclass

import numpy as np

from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_po_manipulation.prm import PRM
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG
from lab_ur_stack.manipulation.utils import to_canonical_config
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2


@dataclass
class ItemPosition:
    name: str
    position: tuple[float, float, float]
    pick_start_offset: tuple[float, float, float]
    pickup_ee_rz: float = 0


item_positions = [
 ItemPosition("top_shelf_left", (-0.777, -0.189, 0.524), pick_start_offset=(0.15, 0, 0)),
 ItemPosition("top_shelf_right", (-0.796, 0.084, -0.524), pick_start_offset=(0.15, 0, 0)),
 ItemPosition("middle_shelf", (-0.781, -0.055, 0.287), pick_start_offset=(0.15, 0, 0)),
]

mp = POManMotionPlanner()
gt = GeometryAndTransforms.from_motion_planner(mp)

r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
r2_controller = ManipulationControllerVG(ur5e_2["ip"], ur5e_2["name"], mp, gt)
r1_controller.speed, r1_controller.acceleration = 0.5, .6
r2_controller.speed, r2_controller.acceleration = 0.5, .6

r1_controller.plan_and_move_home()
r2_controller.plan_and_move_home()

r1_controller.release_grasp()

# test picking up:
item = ItemPosition("test_item", position=(0.35, -0.35, 0.2), pick_start_offset=(-0.15, 0, 0), pickup_ee_rz=np.pi/2)

# Compute approach vector (from offset to target)
offset_position = np.array([
    item.position[0] + item.pick_start_offset[0],
    item.position[1] + item.pick_start_offset[1],
    item.position[2] + item.pick_start_offset[2]
])

default_direciton = np.array([0, 0, 1])
approach_vector = -np.array(item.pick_start_offset)
approach_vector /= np.linalg.norm(approach_vector)
rotation_axis = np.cross(default_direciton, approach_vector)
rotation_angle = np.arccos(np.dot(default_direciton, approach_vector))
axis_angle = rotation_axis * rotation_angle
axis_angle = axis_angle * item.pickup_ee_rz

pick_up_start_pose = offset_position.tolist() + axis_angle.tolist()

pick_up_start_config = r1_controller.getInverseKinematics(pick_up_start_pose)
pick_up_start_config = to_canonical_config(pick_up_start_config)
r1_controller.plan_and_moveJ(pick_up_start_config, speed=0.5, acceleration=0.6)
item_pose = np.concatenate((item.position, axis_angle))
r1_controller.moveL(item_pose, speed=0.1, acceleration=0.6)
r1_controller.grasp()
r1_controller.moveL(pick_up_start_pose, speed=0.1, acceleration=0.6)



# prm = PRM.load_roadmap(mp, "roadmap_ur5e_1.npy")
#
# red_cup_config = [0.16811437904834747,
#  -1.582806249658102,
#  1.9270833174334925,
#  -3.4821816883482875,
#  -1.7212699095355433,
#  0.003565816441550851]
#
# path = prm.find_path(r1_controller.getActualQ(), red_cup_config)
#
# r1_controller.move_path(path, speed=0.3, acceleration=0.6)


pass