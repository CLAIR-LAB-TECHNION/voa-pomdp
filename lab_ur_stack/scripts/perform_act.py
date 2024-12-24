import json
import logging
from typing import Dict, Any
import math

from fontTools.misc.psOperators import ps_string
from klampt import vis
import matplotlib.pyplot as plt
from lab_ur_stack.manipulation.utils import ur5e_2_collect_blocks_from_positions
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                workspace_y_lims_default, goal_tower_position)

from lab_ur_stack.vision.utils import lookat_verangle_horangle_distance_to_robot_config, \
    detections_plots_no_depth_as_image, detections_plots_with_depth_as_image
import numpy as np
import time

int_items_pos = [[-1, -0.4, 0], [-0.8, -0.4, 0], [-1, -0.4, 0], [-1, -0.4, 0]]
int_items_angles = [(0, 0, math.pi / 2), (0, 0, 0)]


initial_positions_mus = [[-0.8, -0.75], [-0.6, -0.65]]
initial_positions_sigmas = [[0.04, 0.02], [0.05, 0.07]]

# fixed help config:
lookat = [np.mean(workspace_x_lims_default), np.mean(workspace_y_lims_default), 0]
lookat[1] +=0.05
# vertical_angle = 35
# horizontal_angle = 20
# distance = 1
vertical_angle = 25
horizontal_angle = 0
distance = 0.7

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_workspace(json_file: str) -> Dict[str, Any]:
    """Loads workspace configuration from a JSON file."""
    try:
        with open(json_file, 'r') as file:
            workspace = json.load(file)
        logging.info("Workspace loaded successfully.")
        return workspace
    except FileNotFoundError:
        logging.error("JSON file not found.")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON file.")
        raise

def initialize_workspace(workspace, mp, gt):
    return

def save_workspace(json_file: str, workspace: Dict[str, Any]):
    """Saves the updated workspace back to the JSON file."""
    try:
        # Use 'ensure_ascii=False' to support non-ASCII characters (if needed)
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(workspace, file, indent=4, ensure_ascii=False)
        logging.info("Workspace saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save workspace: {e}")
        raise

def handle_command(command: Dict[str, str], workspace: Dict[str, Any]) -> str:
    """Handles a command to move an object in the workspace."""
    try:
        action = command.get("action")
        item = command.get("item")
        subsection = command.get("subsection")

        if not action or not item or not subsection:
            raise ValueError("Command must include 'action', 'item', and 'subsection'.")

        if item not in workspace['items']:
            raise ValueError(f"Invalid item '{item}'. Available items: {list(workspace['items'].keys())}.")

        valid_subsections = [sub for subs in workspace['sections'].values() for sub in subs]
        if subsection not in valid_subsections:
            raise ValueError(f"Invalid subsection '{subsection}'. Available subsections: {valid_subsections}.")

        if action == "pick-up":
            if workspace['robot']['holding'] is not None:
                raise ValueError("Robot is already holding an item.")
            if workspace['items'][item] != subsection:
                raise ValueError(f"Item '{item}' is not in subsection '{subsection}'.")
            workspace['robot']['holding'] = item
            workspace['items'][item] = None
            logging.info(f"Picked up {item} from {subsection}.")

        elif action == "put-down":
            if workspace['robot']['holding'] != item:
                raise ValueError(f"Robot is not holding '{item}'.")
            workspace['robot']['holding'] = None
            workspace['items'][item] = subsection
            logging.info(f"Put down {item} in {subsection}.")

        else:
            raise ValueError(f"Unknown action '{action}'.")

        return "Action completed successfully."

    except ValueError as e:
        logging.error(f"Command error: {e}")
        return f"Command failed: {e}"



def test_mp(mp, ws):
    item_name = ws["items"].get("red-can", "unknown")
    location = ws["items"]["red-can"].get("coordinates")
    height = ws["items"]["red-can"].get("height")
    rz = 0  # Default rotation around z-axis
    pick_up_point = location[:]
    pick_up_point[2] += height * 1.1  # Lift up for safety
    # pick_up_config = ManipulationController.find_ik_solution()
    # pick_up_config = [0, 1.5707963267948966, -0.7853981633974483, 0.7853981633974483, 0, 1.0471975511965976, 0, 1.0471975511965976,]
    pick_up_config = [1.5707963267948966, -0.7853981633974483, 0, 0, 1, 0]
    robot = mp.robot_name_mapping["ur5e_2"]
    current_config = robot.getConfig()
    current_config = current_config[1:7]
    mp.plan_from_start_to_goal_config("ur5e_2", current_config, pick_up_config)



def pick_up_item(robot, item):
    """
    Picks up an item based on its metadata from the workspace.

    Parameters:
    - robot: Instance of ManipulationController controlling the robot.
    - item: Dictionary containing item's metadata (name, coordinates, height, etc.).

    Steps:
    1. Extract item's position and height from metadata.
    2. Move to a position above the item safely.
    3. Execute the pick_up function for the final pick-up action.
    """
    # Extract item properties
    item_name = item.get("name", "unknown")
    location = item.get("coordinates")
    height = item.get("height")  # Mandatory field
    rz = 0  # Default rotation around z-axis

    try:

        # Validate mandatory fields
        if not location:
            raise ValueError(f"Item '{item_name}' has no coordinates specified.")
        if height is None:
            raise ValueError(f"Item '{item_name}' is missing a height value. This is critical to avoid collisions.")

        # Calculate the position above the item for a safe approach
        pick_up_point = location[:]
        pick_up_point[2] += height * 1.1  # Lift up for safety

        # Find the IK solution for the above position
        logging.info(f"Calculating IK solution for position above '{item_name}' at {pick_up_point}.")
        pick_up_config = robot.find_ik_solution(pick_up_point)
        if not pick_up_config:
            raise ValueError(f"Failed to find an IK solution for '{item_name}' at {pick_up_point}.")

        # Plan and move to the position above the item
        logging.info(f"Moving to position above '{item_name}' at {pick_up_point}.")
        if not robot.plan_and_moveJ(pick_up_config):
            raise RuntimeError(f"Failed to move to the position above '{item_name}'.")

        # Perform the pick-up action
        logging.info(f"Executing pick-up action for '{item_name}'.")
        robot.pick_up(x=location[0], y=location[1], rz=rz, start_height=height * 1.2)
        robot.mp.remove_object(item["name"])
        robot.mp._add_attachments(robot.mp.world.robot("ur5e_2"), [item["name"]])  # Todo: change the ur5e_2 to the robot name
        robot.mp.visualize(window_name="workspace")

    except Exception as e:
        logging.error(f"Failed to pick up item '{item_name}': {e}")
        raise


def drop_down_item(robot, item, goal):
    """
    Drops down an item at the specified coordinates in the workspace.

    Parameters:
    - robot: Instance of ManipulationController controlling the robot.
    - item: Dictionary containing item's metadata (name, height, etc.).
    - drop_coordinates: List of [x, y, z] coordinates where the item should be placed.

    Steps:
    1. Verify if the gripper is holding an item. If yes, release it.
    2. Move to a position above the drop point safely.
    3. Execute the drop-down action by moving down and releasing the gripper.
    """
    try:
        # Extract item properties
        item_name = item.get("name", "unknown")
        height = item.get("height")  # Mandatory field
        rz = 0  # Default rotation around z-axis

        # Validate mandatory fields
        if height is None:
            raise ValueError(f"Item '{item_name}' is missing a height value. This is critical to avoid collisions.")

        # Check if the gripper is holding an item and release it
        if robot.is_gripper_engaged():  # Assuming is_gripper_engaged() checks if the gripper is holding something
            logging.info("Gripper is engaged. Releasing it before starting the drop-down operation.")
            robot.release_grasp()

        # Calculate the position above the drop point for a safe approach
        above_drop_point = goal[:]
        above_drop_point[2] += height * 1.1  # Lift up for safety

        # Find the IK solution for the above position
        logging.info(f"Calculating IK solution for position above drop point at {above_drop_point}.")
        above_drop_config = robot.find_ik_solution(above_drop_point)
        if not above_drop_config:
            raise ValueError(f"Failed to find an IK solution for position above drop point at {above_drop_point}.")

        # Plan and move to the position above the drop point
        logging.info(f"Moving to position above drop point at {above_drop_point}.")
        if not robot.plan_and_moveJ(above_drop_config):
            raise RuntimeError(f"Failed to move to the position above the drop point.")

        # Move down to the drop point
        logging.info(f"Moving down to the drop point at {goal}.")
        robot.plan_and_move_to_xyzrz(
            x=goal[0],
            y=goal[1],
            z=goal[2],
            rz=rz,
            for_down_movement=True
        )

        # Release the gripper to drop the item
        logging.info(f"Dropping the item '{item_name}' at {goal}.")
        robot.release_grasp()

        # Add the dropped item to the world and dethatch it from the gripper
        mp._add_attachments(mp.world.robot("ur5e_2"), mp.default_attachments["ur5e_2"], color=[0.8, 0.8, 0.8])
        robot.mp.add_object_to_world(item["name"], item)

        # Move back to the above position for safety
        logging.info(f"Moving back to the position above the drop point.")
        robot.plan_and_moveJ(above_drop_config)

    except Exception as e:
        logging.error(f"Failed to drop down item '{item_name}': {e}")
        raise



def main():
    workspace_file = "workspace.json"
    workspace = load_workspace(workspace_file)

    'configuring the robot'
    # camera = RealsenseCamera()
    mp = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(mp)
    # position_estimator = ImageBlockPositionEstimator(workspace_x_lims_default, workspace_y_lims_default, gt)

    # r1_camera = ManipulationController(ur5e_1["ip"], ur5e_1["name"], mp, gt)
    # r2_griper = ManipulationController(ur5e_2["ip"], ur5e_2["name"], mp, gt)
    # r1_camera.speed, r1_camera.acceleration = 0.75, 0.75
    # r2_griper.speed, r2_griper.acceleration = 2, 1.2
    # help_config = lookat_verangle_horangle_distance_to_robot_config(lookat, vertical_angle, horizontal_angle,
    #                                                                 distance, gt, "ur5e_1")

    command = {
        "action": "pick-up",
        "item": "red-can",
        "section": "blue_zone_1"
    }
    red_can = workspace["items"]["red-can"]
    spray_bottle = workspace["items"]["spray-bottle"]
    spray_bottle1 = workspace["items"]["spray-bottle1"]

    mp.add_object_to_world("spray-bottle", spray_bottle)
    mp.add_object_to_world("spray-bottle1", spray_bottle1)
    mp.add_object_to_world("red-can", red_can)
    mp.visualize(window_name="workspace")

    mp._add_attachments(mp.world.robot("ur5e_2"), ["spray-bottle"])
    mp._add_attachments(mp.world.robot("ur5e_2"), ["red-can"])
    mp._add_attachments(mp.world.robot("ur5e_2"), mp.default_attachments["ur5e_2"], color=[0.8, 0.8, 0.8])
    mp.remove_object("spray-bottle1")

    test_mp(mp, workspace)


    # mp.remove_object("spray-bottle")
    status = handle_command(command, workspace)
    print(status)

    save_workspace(workspace_file, workspace)

if __name__ == "__main__":
    main()