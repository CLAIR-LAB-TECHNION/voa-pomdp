import time
import numpy as np
from klampt import WorldModel, Geometry3D, RobotModel
from klampt.model.geometry import box
from klampt import vis
from klampt.plan.cspace import MotionPlan
from klampt.plan.robotcspace import RobotCSpace
from klampt.model.collide import WorldCollider
from klampt.plan import robotplanning
from klampt.model import ik
import os


class MotionPlanner:
    default_attachments = {"u5e_1": ["camera", "gripper"], "u5e_2": ["gripper"]}

    def __init__(self, eps=1e-2, attachments=default_attachments):
        """
        parameters:
        eps: epsilon gap for collision checking along the line in configuration space. Too high value may lead to
            collision, too low value may lead to slow planning. Default value is 1e-2.
        """
        self.eps = eps

        self.world = WorldModel()
        # dir = os.path.dirname(os.path.realpath(__file__))
        # world_path = os.path.join(dir, "klampt_world.xml")
        world_path = "klampt_world.xml"
        self.world.readFile(world_path)


        # self._build_world()

        # self.robot = self.world.robot(0)
        # self.ee_link = self.robot.link("ee_link")

        # values are imported from configuration
        # self.robot.setJointLimits(limits_l, limits_h)
        # self.planning_config = default_config

    def visualize(self):
        """
        open visualization window
        """
        beckend = "PyQt"
        vis.init(beckend)

        vis.add("world", self.world)

        # set camera position:
        viewport = vis.getViewport()
        viewport.camera.tgt = [0, 0, 0.7]
        viewport.camera.rot = [0, -0.7, 0]

        vis.show()


if __name__ == "__main__":
    planner = MotionPlanner()
    planner.visualize()

    time.sleep(300)
