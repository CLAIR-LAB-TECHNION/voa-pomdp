import time
from frozendict import frozendict
import numpy as np
from numpy import pi
from klampt import WorldModel, Geometry3D, RobotModel
from klampt.model.geometry import box
from klampt import vis
from klampt.plan.cspace import MotionPlan
from klampt.plan import robotplanning
from klampt.model import ik
from klampt.math import se3, so3
import os


class MotionPlanner:
    default_attachments = frozendict(ur5e_1=["camera", "gripper"], ur5e_2=["gripper"])
    default_settings = frozendict({# "type": "lazyrrg*",
                                    "type": "rrt*",
                                    "bidirectional": False,
                                    "connectionThreshold": 30.0,
                                    # "perturbationRadius": 0.1,
                                    # "suboptimalityFactor": 1.01,  # only for rrt* and prm*.
                                    # Don't use suboptimalityFactor as it's unclear how that parameter works...
                                    # seems like it's ignored even in rrt*
                                    # "shortcut": True, # only for rrt
                                  })

    def __init__(self, eps=1e-2, attachments=default_attachments, settings=default_settings):
        """
        parameters:
        eps: epsilon gap for collision checking along the line in configuration space. Too high value may lead to
            collision, too low value may lead to slow planning. Default value is 1e-2.
        """
        self.eps = eps

        self.world = WorldModel()
        dir = os.path.dirname(os.path.realpath(__file__))
        world_path = os.path.join(dir, "klampt_world.xml")
        self.world.readFile(world_path)

        self.ur5e_1 = self.world.robot("ur5e_1")
        self.ur5e_2 = self.world.robot("ur5e_2")
        self.robot_name_mapping = {"ur5e_1": self.ur5e_1, "ur5e_2": self.ur5e_2}
        self._add_attachments(self.ur5e_1, attachments["ur5e_1"])
        self._add_attachments(self.ur5e_2, attachments["ur5e_2"])

        self.settings = frozendict(self.default_settings)

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
        vis.setColor(('world', 'ur5e_1'), 0, 1, 1)
        vis.setColor(('world', 'ur5e_2'), 0, 0, 0.5)

        # set camera position:
        viewport = vis.getViewport()
        viewport.camera.tgt = [0, 0, 0]
        viewport.camera.rot = [0, -0.5, 0]

        vis.show()

    def vis_config(self, robot_name, config_):
        """
        Show visualization of the robot in a config
        :param robot_name:
        :param config_:
        :return:
        """
        config = config_.copy()
        if len(config) == 6:
            config = self.config6d_to_klampt(config)
        config = [config]  # There's a bug in visualize config so we just visualize a path of length 1

        vis.add("robot_config", config)
        vis.setColor("robot_config", 0, 1, 0, 0.5)
        vis.setAttribute("robot_config", "robot", robot_name)

    def vis_path(self, robot_name, path_):
        """
        show the path in the visualization
        """
        path = path_.copy()
        if len(path[0]) == 6:
            path = [self.config6d_to_klampt(q) for q in path]

        robot = self.robot_name_mapping[robot_name]
        robot.setConfig(path[0])
        robot_id = robot.id

        # trajectory = RobotTrajectory(robot, range(len(path)), path)
        vis.add("path", path)
        vis.setColor("path", 1, 1, 1, 0.5)
        vis.setAttribute("path", "robot", robot_name)

    def show_point_vis(self, point):
        vis.add("point", point)
        vis.setColor("point", 1, 0, 0, 0.5)

    def plan_from_start_to_goal_config(self, robot_name: str, start_config, goal_config, max_time=15,
                                       max_length_to_distance_ratio=10):
        """
        plan from a start and a goal that are given in 6d configuration space
        """
        start_config_klampt = self.config6d_to_klampt(start_config)
        goal_config_klampt = self.config6d_to_klampt(goal_config)
        robot = self.robot_name_mapping[robot_name]
        path = self._plan_from_start_to_goal_config_klampt(robot, start_config_klampt, goal_config_klampt,
                                                           max_time, max_length_to_distance_ratio)

        return self.path_klampt_to_config6d(path)

    def _plan_from_start_to_goal_config_klampt(self, robot, start_config, goal_config, max_time=15,
                                               max_length_to_distance_ratio=10):
        """
        plan from a start and a goal that are given in klampt 8d configuration space
        """
        robot.setConfig(start_config)

        planner = robotplanning.plan_to_config(self.world, robot, goal_config,
                                               # ignore_collisions=[('keep_out_from_ur3_zone', 'table2')],
                                               # extraConstraints=
                                               **self.settings)
        planner.space.eps = self.eps
        return self._plan(planner, max_time, max_length_to_distance_ratio=max_length_to_distance_ratio)

    def _plan(self, planner: MotionPlan, max_time=15, steps_per_iter=1000, max_length_to_distance_ratio=10):
        """
        find path given a prepared planner, with endpoints already set
        @param planner: MotionPlan object, endpoints already set
        @param max_time: maximum planning time
        @param steps_per_iter: steps per iteration
        @param max_length_to_distance_ratio: maximum length of the pass to distance between start and goal. If there is
            still time, the planner will continue to plan until this ratio is reached. This is to avoid long paths
            where the robot just moves around because non-optimal paths are still possible.
        """
        start_time = time.time()
        path = None
        print("planning motion...", end="")
        while (path is None or self.compute_path_length_to_distance_ratio(path) > max_length_to_distance_ratio) \
                and time.time() - start_time < max_time:
            print(".", end="")
            planner.planMore(steps_per_iter)
            path = planner.getPath()
        print("")
        if path is None:
            print("no path found")
        return path

    def plan_multiple_robots(self):
        # implement if\when necessary.
        # robotplanning.plan_to_config supports list of robots and goal configs
        raise NotImplementedError

    @staticmethod
    def config6d_to_klampt(config):
        """
        There are 8 links in our rob for klampt, some are stationary, actual joints are 1:7
        """
        config_klampt = [0] * 8
        config_klampt[1:7] = config
        return config_klampt

    @staticmethod
    def klampt_to_config6d(config_klampt):
        """
        There are 8 links in our rob for klampt, some are stationary, actual joints are 1:7
        """
        return config_klampt[1:7]

    def path_klampt_to_config6d(self, path_klampt):
        """
        convert a path in klampt 8d configuration space to 6d configuration space
        """
        if path_klampt is None:
            return None
        path = []
        for q in path_klampt:
            path.append(self.klampt_to_config6d(q))
        return path

    def compute_path_length(self, path):
        """
        compute the length of the path
        """
        if path is None:
            return np.inf
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        return length

    def compute_path_length_to_distance_ratio(self, path):
        """ compute the ratio of path length to the distance between start and goal """
        if path is None:
            return np.inf
        start = np.array(path[0])
        goal = np.array(path[-1])
        distance = np.linalg.norm(start - goal)
        length = self.compute_path_length(path)
        return length / distance


    def _add_attachments(self, robot, attachments):
        """
        add attachments to the robot. This is very abstract geometry that should be improved later.
        """
        # TODO: measure and improve
        all_attachments_geom = Geometry3D()
        all_attachments_geom.setGroup()

        if "gripper" in attachments:
            gripper_obj = box(0.09, 0.08, 0.08, center=[+0.04, 0, 0.00])
            gripper_geom = Geometry3D()
            gripper_geom.set(gripper_obj)
            all_attachments_geom.setElement(0, gripper_geom)
        if "camera" in attachments:
            camera_obj = box(0.07, 0.15, 0.1, center=[-0.05, 0, 0.05])
            camera_geom = Geometry3D()
            camera_geom.set(camera_obj)
            all_attachments_geom.setElement(1, camera_geom)

        robot.link("ee_link").geometry().set(all_attachments_geom)


if __name__ == "__main__":
    planner = MotionPlanner()
    planner.visualize()

    print(planner.transform_world_to_robot("ur5e_1", [-1, -1, 0, 0, pi, 0]))

    # path = planner.plan_from_start_to_goal_config("ur5e_1",
    #                                        [pi/2 , 0, 0, 0, 0, 0],
    #                                        [0, -pi/2, 0, -pi/2, 0, 0])
    # planner.show_path_vis("ur5e_1", path)

    # will visualize the path on robot1
    path = planner.plan_from_start_to_goal_config("ur5e_2",
                                           [0, 0, 0, 0, 0, 0],
                                           [0, -pi/2, 0, -pi/2, 0, 0])
    planner.vis_path("ur5e_2", path)

    time.sleep(300)
