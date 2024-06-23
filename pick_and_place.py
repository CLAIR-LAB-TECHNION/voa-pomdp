from typing import List
import typer
from manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2


app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True}
)
def main(pickup_x: float, pickup_y: float, drop_x: float, drop_y: float):
    typer.echo(f"Pickup location: {pickup_x, pickup_y}")
    typer.echo(f"Drop location: {drop_x, drop_y}")

    controller = ManipulationController.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])

    # controller.move_home(speed=0.5, acceleration=0.5)

    controller.pick_up(pickup_x, pickup_y, 0)
    controller.put_down(drop_x, drop_y, 0)


if __name__ == "__main__":
    app()
