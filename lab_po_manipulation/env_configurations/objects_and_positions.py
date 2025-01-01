from dataclasses import dataclass
from typing import Sequence
import json
from pathlib import Path


@dataclass
class ItemPosition:
    name: str
    position: Sequence[float]
    # a grasp is defined by initial offset from position and rotation of the end effector around forward axis
    grasps_offset_rz: Sequence[tuple[Sequence[float], float]]

@dataclass
class ManipulatedObject:
    name: str
    grasp_width: float | None = None


@dataclass
class MobileObstacle:
    name: str
    size: Sequence[float]
    initial_position: Sequence[float]


def save_grasps_to_json(position_name: str, grasps: list[tuple[list[float], float]],
                        output_dir: str = "grasps"):
    dir_path = Path(__file__).parent / output_dir
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path/ f"{position_name}_grasps.json"

    data = {
        "position_name": position_name,
        "grasps": [{"offset": list(g[0]), "rz": g[1]} for g in grasps]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_grasps_from_json(position_name: str,
                          input_dir: str = "grasps")\
        -> list[tuple[list[float], float]]:

    dir_path = Path(__file__).parent / input_dir
    filepath = dir_path/ f"{position_name}_grasps.json"

    # handle failure to load file
    if not filepath.exists():
        print(f"\033[93m Warnning: File {filepath} does not exist \033[0m")
        return []
    with open(filepath) as f:
        data = json.load(f)
        return [(g["offset"], g["rz"]) for g in data["grasps"]]


positions = [
 ItemPosition("top_shelf_left", (-0.796, -0.189, 0.524), []),
 ItemPosition("top_shelf_right", (-0.796, 0.084, 0.524), []),
 ItemPosition("middle_shelf", (-0.781, -0.055, 0.287), []),
]
for pos in positions:
    pos.grasps_offset_rz = load_grasps_from_json(pos.name)
positions_dict = {pos.name: pos for pos in positions}


objects = [
    ManipulatedObject("red_cup"),
    ManipulatedObject("blue_cup"),
    ManipulatedObject("soda_can", grasp_width=57),
]
objects_dict = {obj.name: obj for obj in objects}


mobile_obstacles = [
    MobileObstacle("mobile_obstacle_1", (0.12, 0.12, 0.48), initial_position=(-0.16, -0.36, 0.24)),
    MobileObstacle("mobile_obstacle_2", (0.16, 0.32, 0.08), initial_position=(-0.455, -0.04, 0.28)),
    MobileObstacle("mobile_obstacle_3", (0.12, 0.12, 0.24), initial_position=(-0.685, -0.36, 0.12)),
    MobileObstacle("mobile_obstacle_4", (0.32, 0.08, 0.16), initial_position=(-0.785, -0.1, 0.80)),
]
mobile_obstacles_dict = {mob.name: mob for mob in mobile_obstacles}