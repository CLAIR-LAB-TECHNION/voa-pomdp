from lab_po_manipulation.env_configurations.objects_and_positions import (positions, objects_dict,
                                                                          mobile_obstacles_dict, mobile_obstacles)
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_ur_stack.manipulation.manipulation_controller_vg import ManipulationControllerVG
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from klampt.math import so3, se3


mp = POManMotionPlanner()
gt = GeometryAndTransforms.from_motion_planner(mp)

for mob_obs in mobile_obstacles:
    obj = {"name": mob_obs.name, "scale": mob_obs.size, "coordinates": mob_obs.initial_position,
           "color": [0.8, 0.3, 0.3, 0.75], "geometry_file": "../lab_ur_stack/motion_planning/objects/cube.off",
           "angle": so3.identity()}
    mp.add_object_to_world(mob_obs.name, obj)
mp.visualize()
pass
#
# r1_controller = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
# r2_controller = ManipulationControllerVG(ur5e_2["ip"], ur5e_2["name"], mp, gt)
# r1_controller.speed, r1_controller.acceleration = 0.5, .6
# r2_controller.speed, r2_controller.acceleration = 0.5, .6
#
# # r1_controller.plan_and_move_home()
# r2_controller.plan_and_move_home()
#
# item = positions[0]
# pos = list(item.position)
# pos[0] += 0.
# width = objects_dict["soda_can"].grasp_width
# r1_controller.pick_up_at_angle(pos, item.pick_start_offset, grasp_width=width)
#
# item = positions[2]
# r1_controller.put_down_at_angle(item.position, item.pick_start_offset)

# prm = PRM.load_roadmap(mp, "roadmap_ur5e_1.npy")

# path = prm.find_path(r1_controller.getActualQ(), red_cup_config)
#
# r1_controller.move_path(path, speed=0.3, acceleration=0.6)


pass