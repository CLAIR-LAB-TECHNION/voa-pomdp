from klampt import *
from klampt.plan import robotcspace
from klampt.plan import cspace, robotplanning
import numpy as np
import time

from klampt.plan.robotplanning import make_space

from lab_ur_stack.motion_planning.motion_planner import MotionPlanner


class RoadmapBuilder:
    def __init__(self, motion_planner, robot_name, n_samples=1000, k_neighbors=10, max_edge_distance=2.0, eps=1e-2):
        """
        Build and manage PRM roadmap for a robot
        Args:
            motion_planner: Instance of MotionPlanner class
            robot_name: Name of robot to build roadmap for
            n_samples: Number of random configurations to sample
            k_neighbors: Number of nearest neighbors to try connecting
            max_edge_distance: Maximum allowed distance between configs to attempt connection
        """
        self.mp = motion_planner
        self.robot_name = robot_name
        self.robot = self.mp.robot_name_mapping[robot_name]

        self.space = make_space(self.mp.world, self.robot)

        # workaround to set space where visibility check works:
        self.robot.setConfig(self.mp.config6d_to_klampt([0, -np.pi/2, 0, -np.pi/2, 0, 0]))
        c = self.mp.config6d_to_klampt([-np.pi/3]*6)
        planner = robotplanning.plan_to_config(self.mp.world, self.robot, c, **self.mp.settings)
        self.space_for_visibility = planner.space
        self.space_for_visibility.eps = eps

        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        self.max_edge_distance = max_edge_distance

        # Roadmap storage
        self.vertices = []
        self.edges = []

    def sample_random_config(self):
        """Generate random 6D configuration using CSpace sampler"""
        while True:
            q = self.space.sample()
            q = self.mp.klampt_to_config6d(q)
            if self.mp.is_config_feasible(self.robot_name, q):
                return q

    def build_roadmap(self):
        """Build the PRM roadmap by sampling and connecting vertices"""
        print(f"Building roadmap with {self.n_samples} samples...")

        # Sample valid configurations
        for i in range(self.n_samples):
            if i % 100 == 0:
                # print on top of prev line
                print(f"Sampling vertex {i}/{self.n_samples}".ljust(50), end="\r")
            config = self.sample_random_config()
            self.vertices.append(config)
        print()
        print("Connecting vertices...")

        # Convert vertices list to numpy array for faster operations
        vertices_array = np.array(self.vertices)

        print("Connecting vertices...")
        num_attempted = 0
        num_successful = 0

        n_vertices = len(self.vertices)
        diff = vertices_array[:, np.newaxis, :] - vertices_array[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        for i in range(n_vertices):
            if i % 10 == 0:
                print(f"Processing vertex {i}/{n_vertices}".ljust(75), end="\r")

            dist_to_others = distances[i]
            dist_to_others[i] = float('inf')  # Exclude self
            neighbor_indices = np.argpartition(dist_to_others, self.k_neighbors)[:self.k_neighbors]

            for j in neighbor_indices:
                dist = distances[i, j]
                if dist > self.max_edge_distance:
                    continue

                num_attempted += 1
                if self.check_edge_validity(self.vertices[i], self.vertices[j]):
                    self.edges.append((i, j))
                    num_successful += 1

        print()
        print(f"Roadmap built with {len(self.vertices)} vertices and {len(self.edges)} edges")
        print(f"Attempted {num_attempted} connections, {num_successful} successful")

    def check_edge_validity(self, config1, config2):
        """Check if direct path between two configs is valid using CSpace"""
        config1_klampt = self.mp.config6d_to_klampt(config1)
        self.robot.setConfig(config1_klampt)
        result = self.space_for_visibility.isVisible(config1, config2)
        return result

    def save_roadmap(self, filename):
        """Save roadmap to file"""
        data = {
            'vertices': self.vertices,
            'edges': self.edges,
            'robot_name': self.robot_name,
            'params': {
                'n_samples': self.n_samples,
                'k_neighbors': self.k_neighbors,
                'max_edge_distance': self.max_edge_distance
            }
        }
        np.save(filename, data)

    def load_roadmap(self, filename):
        """Load roadmap from file"""
        data = np.load(filename, allow_pickle=True).item()
        assert data['robot_name'] == self.robot_name, "Loaded roadmap is for a different robot"
        self.vertices = data['vertices']
        self.edges = data['edges']
        print(f"Loaded roadmap with {len(self.vertices)} vertices and {len(self.edges)} edges")

    def visualize_roadmap(self):
        """Visualize the roadmap in Klampt viewer"""
        # Visualize vertices
        for i, config in enumerate(self.vertices):
            name = f"vertex_{i}"
            self.mp.vis_config(self.robot_name, config, vis_name=name, rgba=(0, 0, 1, 0.3))

    def visualize_roadmap_ee_poses(self):
        """Visualize the roadmap in Klampt viewer"""
        # Visualize vertices
        for i, config in enumerate(self.vertices):
            name = f"vertex_{i}"
            ee_pose = self.mp.get_forward_kinematics(self.robot_name, config)
            self.mp.show_point_vis(ee_pose[1], name = f"{i}")


if __name__ == "__main__":
    planner = MotionPlanner()
    # planner.visualize()

    builder = RoadmapBuilder(planner, "ur5e_1", n_samples=1000, k_neighbors=10, max_edge_distance=20.)

    builder.build_roadmap()

    builder.save_roadmap("roadmap_ur5e_1.npy")

    # Visualize roadmap
    # builder.visualize_roadmap()
    builder.visualize_roadmap_ee_poses()
    print("n_vertices:", len(builder.vertices))
    print("n_edges:", len(builder.edges))

    # Keep window open
    time.sleep(300)

