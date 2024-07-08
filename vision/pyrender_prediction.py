import json
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from camera.configurations_and_params import color_fx, color_fy, color_ppx, color_ppy


def load_data():
    metadata = json.load(open("images_data_merged_hires/merged_metadata.json", "r"))
    robot_configs = []
    images = []
    depth_images = []
    actual_block_positions = []

    for im_metadata in metadata:
        robot_configs.append(im_metadata["ur5e_1_config"])

        image_path = f"images_data_merged_hires/images/{im_metadata['image_rgb']}"
        image = np.load(image_path)
        images.append(image)

        depth_images.append(np.load(f"images_data_merged_hires/depth/{im_metadata['image_depth']}"))

    # actual block positions are the same for all images, so we can just take the first one
    actual_block_positions = metadata[0]["block_positions"]

    return robot_configs, images, depth_images, actual_block_positions


class PyrenderRenderer:
    def __init__(self, gt: GeometryAndTransforms = None):
        self.gt = gt if gt is not None else GeometryAndTransforms.build()
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0, 1.0])
        self.renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)

        camera = pyrender.IntrinsicsCamera(color_fx, color_fy, color_ppx, color_ppy)
        self.scene.add(camera)

        point_light_positions = [
            [2.0, 2.0, 3.0],
            [-2.0, -2.0, 3.0],
            [2.0, -2.0, 3.0],
            [-2.0, 2.0, 3.0]]
        for pos in point_light_positions:
            light_pose = np.eye(4)
            light_pose[:3, 3] = pos
            light = pyrender.PointLight(color=np.ones(3), intensity=500.0)
            self.scene.add(light, pose=light_pose)

    def set_boxes_positions(self, positions):
        # remove all boxes from the scene
        for node in self.scene.mesh_nodes:
            self.scene.remove_node(node)

        for coord in positions:
            box_mesh = trimesh.creation.box(extents=box_size)
            box_mesh.visual.face_colors = [0, 255, 0, 255]
            box_mesh.apply_translation(coord)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 1.0, 0.0, 1.0])
            box = pyrender.Mesh.from_trimesh(box_mesh, material=material)
            self.scene.add(box)

    def render_from_robot_config(self, robot_name, robot_config):
        camera_pose = self.gt.camera_to_world_transform(robot_name, robot_config)
        camera_pose = self.gt.se3_to_4x4(camera_pose)
        color, depth = self.render_from_camera_pose(camera_pose)
        return color, depth

    def render_from_camera_pose(self, camera_pose_4x4):
        # update camera pose:
        camera = self.scene.main_camera_node
        camera.matrix = self.camera_pose_to_openGL(camera_pose_4x4)
        return self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)

    def camera_pose_to_openGL(self, camera_pose):
        # y and z are in the opposite directions
        opengl_camera_transform = np.array([[1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, -1, 0],
                                            [0, 0, 0, 1]])
        return camera_pose @ opengl_camera_transform


box_size = [0.04, 0.04, 0.04]
plane_size = [0.84, 1.85, 0.01]
plane_position = [-0.805, -0.615, 0]

if __name__ == "__main__":
    robot_configs, images, depth_images, actual_block_positions = load_data()

    box_3d_positons = [(p[0], p[1], 0.02) for p in actual_block_positions]

    r = PyrenderRenderer()
    r.set_boxes_positions(box_3d_positons)

    for robot_config, image in zip(robot_configs, images):
        rendered_color, depth = r.render_from_robot_config("ur5e_1", robot_config)

        # Extract the green boxes and black edges
        mask = rendered_color[:, :, 3] > 0  # Mask where alpha > 0
        overlay = np.zeros_like(rendered_color)
        overlay[mask] = rendered_color[mask]

        # plot rendered_color on top of image, treat transparent background of rendered as transparent
        plt.imshow(image)
        plt.imshow(overlay, alpha=0.5)
        plt.axis('off')
        plt.show()
