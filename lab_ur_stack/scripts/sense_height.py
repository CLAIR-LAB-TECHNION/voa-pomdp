import typer
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_2


app = typer.Typer()

point1 = [-0.54105, -0.95301]
point2 = [-0.64105, -0.95301]
point3 = [-0.59105, -0.95301]
points = [point1, point2, point3]

@app.command(
    context_settings={"ignore_unknown_options": True})
def main(repeat: int = 1):
    controller = ManipulationController.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])
    controller.move_home()

    for point in points:
        heights = []
        print("---tilted:")
        for i in range(repeat):
            h = controller.sense_height_tilted(point[0], point[1])
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

