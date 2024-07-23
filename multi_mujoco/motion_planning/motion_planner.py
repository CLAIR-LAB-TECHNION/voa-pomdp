import os
from frozendict import frozendict
import time

from klampt import WorldModel, Geometry3D, vis
from klampt.math import se3
from klampt.model import collide, ik
from klampt.model.geometry import box
from klampt.plan.cspace import MotionPlan
from klampt.plan import robotplanning

from configurations import *


class MotionPlanner:

    def __init__(self, eps=1e-1):
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
        self._build_world()

        self.robot_0 = self.world.robot("ur5e_1")
        self.robot_0.setJointLimits(limits_l, limits_h)
        self.robot_1 = self.world.robot("ur5e_2")
        self.robot_1.setJointLimits(limits_l, limits_h)
        self.robot_name_mapping = {"ur5e_1": self.robot_0, "ur5e_2": self.robot_1}

        self.world_collider = collide.WorldCollider(self.world)

        self.planning_config = frozendict(default_config)

    def set_config(self, configs):
        for robot_name, config in configs.items():
            self.robot_name_mapping[robot_name].setConfig(config)

    def add_block(self, name, position, color=(0.3, 0.3, 0.3, 0.8)):
        """
        add block to the world
        """
        self._add_box_geom(name, block_size, position, color)

    def move_block(self, name, position):
        """
        move block to position
        """
        rigid_obj = self.world.rigidObject(name)
        width, depth, height = block_size
        box_obj = box(width=width, height=height, depth=depth, center=position)
        rigid_obj.geometry().set(box_obj)

    def attach_box_to_ee(self):
        """
        attach a box to the end effector for collision detection. Should be called once
        """
        # Note that the order is different here, width is in z direction
        sx, sy, sz = block_size
        box_obj = box(width=sz, height=sy, depth=sx, center=[grasp_offset, 0, 0])
        box_geom = Geometry3D()
        box_geom.set(box_obj)

        self.ee_link.geometry().set(box_geom)

    def detach_box_from_ee(self):
        """
        detach the box from the end effector
        """
        dummy_box_obj = box(width=0.001, height=0.001, depth=0.001, center=[0, 0, 0])
        dummy_box_geom = Geometry3D()
        dummy_box_geom.set(dummy_box_obj)

        self.ee_link.geometry().set(dummy_box_geom)

    def _build_world(self):
        """ build the obstacles in the world """
        # Floor
        self._add_box_geom("floor", (0, 0, .05), [0, 0, 0], [0.1, 0.1, 0.1, 1], False)

        # Tables
        table_left_size = [.74, 1.2, .02]
        table_left_pos = [-.6, -.6, 0]
        self._add_box_geom("table_left", table_left_size, table_left_pos, [0.5, 0.5, 0.5, 0.8], False)

        table_right_size = [.6, .6, .02]
        table_right_pos = [.2, 0, 0]
        self._add_box_geom("table_right", table_right_size, table_right_pos, [0.5, 0.5, 0.5, 0.8], False)

        # # Walls
        # wall1_size = [2.44, 0.01, 2.0]
        # wall1_pos = [0, 0.36, 1.0]
        # self._add_box_geom("wall1", wall1_size, wall1_pos, [0.5, 0.5, 0.5, 1], False)
        #
        # wall2_1_size = [0.01, 2.07, 2.0]
        # wall2_1_pos = [-1.165, -0.615, 1.0]
        # self._add_box_geom("wall2_1", wall2_1_size, wall2_1_pos, [0.5, 0.5, 0.5, 1], False)
        #
        # wall2_2_size = [1.0, 0.01, 2.0]
        # wall2_2_pos = [-0.805, -1.90, 1.0]
        # self._add_box_geom("wall2_2", wall2_2_size, wall2_2_pos, [0.5, 0.5, 0.5, 1], False)
        #
        # # UR3e keep-out zone
        # ur3e_zone_size = [0.5, 0.5, 2.0]
        # ur3e_zone_pos = [-0.725, -0.007, 0.50]
        # self._add_box_geom("keep_out_from_ur3_zone", ur3e_zone_size, ur3e_zone_pos, [0.5, 0.5, 0.5, 1], False)

    def _add_box_geom(self, name, size, center, color, update_vis=True):
        """
        add box geometry for collision in the world
        """
        width, depth, height = size
        box_obj = box(width=width, height=height, depth=depth, center=center)
        box_geom = Geometry3D()
        box_geom.set(box_obj)
        box_rigid_obj = self.world.makeRigidObject(name)
        box_rigid_obj.geometry().set(box_geom)
        box_rigid_obj.appearance().setColor(*color)

        if update_vis:
            vis.add("world", self.world)

    def visualize(self, backend="GLUT"):
        """
        open visualization window
        """
        vis.init(backend)

        vis.add("world", self.world)
        # vis.setColor(('world', 'ur5e_1'), 0, 1, 1)
        # vis.setColor(('world', 'ur5e_2'), 0, 0, 0.5)

        # set camera position:
        viewport = vis.getViewport()
        viewport.camera.tgt = [0, -0.6, 0.5]
        viewport.camera.rot = [0, -0.75, 1]
        viewport.camera.dist = 5

        vis.show()

    def vis_config(self, robot_name, config_, vis_name="robot_config", rgba=(0, 0, 1, 0.5)):
        """
        Show visualization of the robot in a config
        :param robot_name:
        :param config_:
        :param rgba: color and transparency
        :return:
        """
        config = config_.copy()
        if len(config) == 6:
            config = self.config6d_to_klampt(config)
        config = [config]  # There's a bug in visualize config so we just visualize a path of length 1

        vis.add(vis_name, config)
        vis.setColor(vis_name, *rgba)
        vis.setAttribute(vis_name, "robot", robot_name)

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

    def show_point_vis(self, point, name="point"):
        vis.add(name, point)
        vis.setColor(name, 1, 0, 0, 0.5)

    def show_ee_poses_vis(self):
        """
        show the end effector poses of all robots in the
        """
        for robot in self.robot_name_mapping.values():
            ee_transform = robot.link("ee_link").getTransform()
            vis.add(f"ee_pose_{robot.getName()}", ee_transform)

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
                                               **self.planning_config)
        planner.space.eps = self.eps

        # before planning, check if a direct path is possible, then no need to plan
        if self._is_direct_path_possible(planner, start_config, goal_config):
            return [goal_config]

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
        print("planning took ", time.time() - start_time, " seconds.")
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
        if config_klampt is None:
            return None
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

    def _is_direct_path_possible(self, planner, start_config_, goal_config_):
        # EmbeddedRobotCspace only works with the active joints:
        start_config = self.klampt_to_config6d(start_config_)[:3]
        goal_config = self.klampt_to_config6d(goal_config_)[:3]
        return planner.space.isVisible(start_config, goal_config)

    def is_config_feasible(self, robot_name, config):
        """
        check if the config is feasible (not within collision)
        """
        if len(config) == 6:
            config_klampt = self.config6d_to_klampt(config)
        else:
            config_klampt = config.copy()

        robot = self.robot_name_mapping[robot_name]
        current_config = robot.getConfig()
        robot.setConfig(config_klampt)

        # we have to get all collisions since there is no method for robot-robot collisions-+--
        all_collisions = list(self.world_collider.collisions())

        robot.setConfig(current_config)  # return to original motion planner state

        # all collisions is a list of pairs of colliding geometries. Filter only those that contains a name that
        # ends with "link" and belongs to the robot, and it's not the base link that always collides with the table.
        for g1, g2 in all_collisions:
            if g1.getName().endswith("link") and g1.getName() != "base_link" and g1.robot().getName() == robot_name:
                return False
            if g2.getName().endswith("link") and g2.getName() != "base_link" and g2.robot().getName() == robot_name:
                return False

        return True

    def get_forward_kinematics(self, robot_name, config):
        """
        get the forward kinematics of the robot, this already returns the transform to world!
        """
        if len(config) == 6:
            config_klampt = self.config6d_to_klampt(config)
        else:
            config_klampt = config.copy()

        robot = self.robot_name_mapping[robot_name]

        previous_config = robot.getConfig()
        robot.setConfig(config_klampt)
        link = robot.link("ee_link")
        ee_transform = link.getTransform()
        robot.setConfig(previous_config)

        return ee_transform

    def ik_solve(self, robot_name, ee_transform, start_config=None):

        if start_config is not None and len(start_config) == 6:
            start_config = self.config6d_to_klampt(start_config)

        robot = self.robot_name_mapping[robot_name]
        return self.klampt_to_config6d(self._ik_solve_klampt(robot, ee_transform, start_config))

    def _ik_solve_klampt(self, robot, ee_transform, start_config=None):

        curr_config = robot.getConfig()
        if start_config is not None:
            robot.setConfig(start_config)

        ik_objective = ik.objective(robot.link("ee_link"), R=ee_transform[0], t=ee_transform[1])
        res = ik.solve(ik_objective, tol=1e-5)
        if not res:
            print("ik not solved")
            robot.setConfig(curr_config)
            return None

        res_config = robot.getConfig()

        robot.setConfig(curr_config)

        ik_objective = ik.IKObjective()

        return res_config


if __name__ == "__main__":
    planner = MotionPlanner()
    planner.visualize(backend="PyQt5")

    point = [0, 0, 1]
    transform = planner.get_forward_kinematics("robot_0", planner.robot_0.getConfig()[1:7])

    point_transofrmed = se3.apply(transform, point)
    planner.show_point_vis(point_transofrmed)
    print("point transformed: ", point_transofrmed)
    # planner.is_config_feasible("robot_0", [0, 0, 0, 0, 0, 0])
    #
    # path = planner.plan_from_start_to_goal_config("robot_0",
    #                                               [np.pi / 2, 0, 0, 0, 0, 0],
    #                                               [0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
    # planner.vis_path("robot_0", path)
    #
    # # will visualize the path on robot1
    # path = planner.plan_from_start_to_goal_config("robot_1",
    #                                        [0, 0, 0, 0, 0, 0],
    #                                        [0, -np.pi/2, 0, -np.pi/2, 0, 0])
    # planner.vis_path("robot_1", path)

    time.sleep(300)
