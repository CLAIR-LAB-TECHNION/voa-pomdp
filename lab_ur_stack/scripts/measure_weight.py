import time

import numpy as np
import typer
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_2


app = typer.Typer()
point = [-0.54105, -0.95301]


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(x: float = point[0],
         y: float = point[1],
         repeat: int = 1):
    controller = ManipulationController.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])

    controller.zeroFtSensor()
    controller.pick_up(x, y, 0)
    time.sleep(0.5)

    ft = controller.getActualTCPForce()
    print("Force:", ft)

    mass = np.sqrt(ft[0]**2 + ft[1]**2 + ft[2]**2) / 9.82
    print("Mass:", mass)

    controller.put_down(x, y, 0)
    controller.grasp()
    time.sleep(0.5)

    ft = controller.getActualTCPForce()
    print("Force:", ft)

    mass = np.sqrt(ft[0]**2 + ft[1]**2 + ft[2]**2) / 9.82
    print("Mass:", mass)


if __name__ == "__main__":
    app()

