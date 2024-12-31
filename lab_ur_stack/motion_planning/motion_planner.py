import os
from klampt.math import se3, so3
from klampt import WorldModel, Geometry3D, RobotModel, vis
from klampt.model.geometry import box
from lab_ur_stack.motion_planning.abstract_motion_planner import AbstractMotionPlanner


class MotionPlanner(AbstractMotionPlanner):
    def _get_klampt_world_path(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "klampt_world.xml")
        return world_path

    def _add_attachments(self, robot, attachments, color=None):
        """
        add attachments to the robot. This is very abstract geometry that should be improved later.
        """
        all_attachments_geom = Geometry3D()
        all_attachments_geom.setGroup()

        element = 0

        if "gripper" in attachments:
            gripper_obj = box(0.09, 0.09, 0.15, center=[0, 0, 0.075])
            gripper_geom = Geometry3D()
            gripper_geom.set(gripper_obj)
            all_attachments_geom.setElement(element, gripper_geom)
            element += 1
            if color is not None:
                vis.setColor(('world', 'ur5e_2', 'ee_link'), color[0], color[1], color[2])

        if "camera" in attachments:
            camera_obj = box(0.18, 0.11, 0.06, center=[0, -0.05, 0.01])
            camera_geom = Geometry3D()
            camera_geom.set(camera_obj)
            all_attachments_geom.setElement(element, camera_geom)
            element += 1

        if "spray-bottle" in attachments:
            spray_bottle_obj = box(0.0625, 0.1, 0.3, center=[0, 0, 0.15])
            spray_bottle_geom = Geometry3D()
            spray_bottle_geom.set(spray_bottle_obj)
            all_attachments_geom.setElement(element, spray_bottle_geom)
            element += 1
            vis.setColor(('world', 'ur5e_2', 'ee_link'), 0, 0, 1)


        if "red-can" in attachments:
            red_can_obj = box(0.05, 0.05, 0.15, center=[0, 0, 0.07])
            red_can_geom = Geometry3D()
            red_can_geom.set(red_can_obj)
            all_attachments_geom.setElement(element, red_can_geom)
            element += 1
            vis.setColor(('world', 'ur5e_2', 'ee_link'), 1, 0, 0)

        if "water-bottle" in attachments:
            water_bottle_obj = box(0.05, 0.05, 0.15, center=[0, 0, 0.07])
            water_bottle_geom = Geometry3D()
            water_bottle_geom.set(water_bottle_obj)
            all_attachments_geom.setElement(element, water_bottle_geom)
            element += 1
            vis.setColor(('world', 'ur5e_2', 'ee_link'), 0.9, 0.9, 0.9)

        if "green-mug" in attachments:
            green_mug_obj = box(0.05, 0.05, 0.15, center=[0, 0, 0.07])
            green_mug_geom = Geometry3D()
            green_mug_geom.set(green_mug_obj)
            all_attachments_geom.setElement(element, green_mug_geom)
            element += 1
            vis.setColor(('world', 'ur5e_2', 'ee_link'), 0, 1, 0)

        # add safety box around where the tool cable is attached
        safety_box = box(0.13, 0.13, 0.04, center=[0, 0, 0.02])
        safety_box_geom = Geometry3D()
        safety_box_geom.set(safety_box)
        all_attachments_geom.setElement(element, safety_box_geom)

        # the positions of tools were measured for ee_offset = 0. move them back by ee_offset
        for i in range(all_attachments_geom.numElements()):
            element = all_attachments_geom.getElement(i)
            # x is forward in ff frame. nothing makes sense anymore...
            element.transform(so3.identity(), [0, 0, -self.ee_offset])
            all_attachments_geom.setElement(i, element)

        robot.link("ee_link").geometry().set(all_attachments_geom)


if __name__ == "__main__":
    planner = MotionPlanner()
    planner.visualize(backend="PyQt5")

    point = [0, 0, 1]
    transform = planner.get_forward_kinematics("ur5e_1", planner.ur5e_1.getConfig()[1:7])

    point_transformed = se3.apply(transform, point)
    planner.show_point_vis(point_transformed)
    print("point transformed: ", point_transformed)
    # planner.is_config_feasible("ur5e_1", [0, 0, 0, 0, 0, 0])

    # path = planner.plan_from_start_to_goal_config("ur5e_1",
    #                                        [pi/2 , 0, 0, 0, 0, 0],
    #                                        [0, -pi/2, 0, -pi/2, 0, 0])
    # planner.show_path_vis("ur5e_1", path)

    # will visualize the path on robot1
    # path = planner.plan_from_start_to_goal_config("ur5e_2",
    #                                        [0, 0, 0, 0, 0, 0],
    #                                        [0, -pi/2, 0, -pi/2, 0, 0])
    # planner.vis_path("ur5e_2", path)

    time.sleep(300)
