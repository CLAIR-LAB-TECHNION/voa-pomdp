import time

import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation.manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2


app = typer.Typer()

point = [-0.54105, -0.95301]


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(x: float = point[0],
         y: float = point[1],
         repeat: int = 2):
    controller = ManipulationController.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])

    time.sleep(1)

    heights = []
    print("---tilted:")
    for i in range(repeat):
        h = controller.sense_height_tilted(x, y)
        print("measured height:", h)
        heights.append(h)

    mean = sum(heights) / len(heights)
    variance = sum((h - mean) ** 2 for h in heights) / len(heights)
    print("measured heights:", heights)
    print("mean:", mean, "variance:", variance)



    # print("-----------------------------")
    # print("---not tilted:")
    # heights = []
    # for i in range(5):
    #     h = controller.sense_height(x, y)
    #     heights.append(h)
    #
    # mean = sum(heights) / len(heights)
    # variance = sum((h - mean) ** 2 for h in heights) / len(heights)
    # print("measured heights:", heights)
    # print("mean:", mean, "variance:", variance)


if __name__ == "__main__":
    app()

