import os
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
import time


class POManMotionPlanner(MotionPlanner):
    def _get_klampt_world_path(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "stat_object_klampt_world.xml")
        return world_path


if __name__ == "__main__":
    planner = POManMotionPlanner()
    planner.visualize()

    time.sleep(100)