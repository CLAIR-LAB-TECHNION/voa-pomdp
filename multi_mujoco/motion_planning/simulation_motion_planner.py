import os

from lab_ur_stack.motion_planning.abstract_motion_planner import AbstractMotionPlanner


class SimulationMotionPlanner(AbstractMotionPlanner):
    def _get_klampt_world_path(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "klampt_world.xml")
        return world_path

    def _add_attachments(self, robot, attachments):
        pass