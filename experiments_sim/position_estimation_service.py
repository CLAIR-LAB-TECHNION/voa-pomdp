import threading
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import BaseManager

from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default


class PositionEstimatorServer:
    """The actual server that runs in its own process"""

    def __init__(self):
        # Initialize estimator
        dummy_env = BlockStackingSimulator(render_mode=None, visualize_mp=False)
        gt = GeometryAndTransforms(dummy_env.motion_executor.motion_planner,
                                   cam_in_ee=-np.array(dummy_env.helper_camera_translation_from_ee))
        self.position_estimator = ImageBlockPositionEstimator(
            workspace_x_lims_default, workspace_y_lims_default,
            gt, "ur5e_1", dummy_env.mujoco_env.get_robot_cam_intrinsic_matrix())

        self.lock = threading.Lock()

    def estimate_position(self, image, robot_config):
        """Thread-safe estimation"""
        with self.lock:
            try:
                # Ensure contiguous arrays
                image = np.ascontiguousarray(image, dtype=np.uint8)
                robot_config = np.ascontiguousarray(robot_config, dtype=np.float64)

                positions, annotations = self.position_estimator.get_block_position_plane_projection(
                    image, robot_config, plane_z=-0.02, return_annotations=True,
                    detect_on_cropped=True, max_detections=5
                )

                # Ensure all parts of result are contiguous arrays
                positions_con = np.ascontiguousarray(positions, dtype=np.float64)
                annotations_con = (np.ascontiguousarray(annotations[0], dtype=np.float64),
                                   np.ascontiguousarray(annotations[1], dtype=np.float64))

                return positions_con, annotations_con
            except Exception as e:
                print(f"Error in position estimation: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

    def shutdown(self):
        pass  # Nothing to clean up


class PositionEstimatorManager(BaseManager):
    pass


# Register the server with the manager
PositionEstimatorManager.register('PositionEstimatorServer', PositionEstimatorServer)


def create_position_estimation_service():
    """Create and start the manager"""
    manager = PositionEstimatorManager()
    manager.start()
    return manager.PositionEstimatorServer()