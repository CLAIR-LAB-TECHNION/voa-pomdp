import multiprocessing as mp
from queue import Empty

from lab_ur_stack.camera.configurations_and_params import color_camera_intrinsic_matrix
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator


class SharedImageBlockPositionEstimator(ImageBlockPositionEstimator):
    def __init__(self, workspace_limits_x, workspace_limits_y, gt: GeometryAndTransforms, robot_name='ur5e_1',
                 intrinsic_camera_matrix=color_camera_intrinsic_matrix):
        # Store only picklable parameters
        self.workspace_limits_x = workspace_limits_x
        self.workspace_limits_y = workspace_limits_y
        self.robot_name = robot_name
        self.intrinsic_camera_matrix = intrinsic_camera_matrix

        # Store parameters needed to recreate GeometryAndTransforms
        self.motion_planner = gt.motion_planner
        self.cam_in_ee = gt.cam_in_ee

        # Setup multiprocessing components
        self.request_queue = mp.Queue()
        self.response_queues = {}

        # Create actual position estimator in the service process
        self.service_process = mp.Process(target=self._service_loop)
        self.service_process.start()

    def _service_loop(self):
        # Recreate GeometryAndTransforms in the service process
        gt = GeometryAndTransforms(self.motion_planner, cam_in_ee=self.cam_in_ee)

        # Create the actual position estimator here in the service process
        position_estimator = ImageBlockPositionEstimator(
            self.workspace_limits_x,
            self.workspace_limits_y,
            gt,
            self.robot_name,
            self.intrinsic_camera_matrix
        )

        while True:
            try:
                client_id, response_queue, params = self.request_queue.get(timeout=1)
                if params is None:
                    break

                result = position_estimator.get_block_position_plane_projection(
                    params['image'],
                    params['robot_config'],
                    plane_z=params.get('plane_z', -0.02),
                    return_annotations=params.get('return_annotations', True),
                    detect_on_cropped=params.get('detect_on_cropped', True),
                    max_detections=params.get('max_detections', 5)
                )
                response_queue.put(result)

            except Empty:
                continue
            except Exception as e:
                print(f"Error in position estimator service: {e}")
                response_queue.put(None)

    def get_block_position_plane_projection(self, images, robot_configurations, plane_z=-0.02,
                                            return_annotations=True, detect_on_cropped=True, max_detections=5):
        client_id = id(mp.current_process())
        if client_id not in self.response_queues:
            self.response_queues[client_id] = mp.Queue()

        params = {
            'image': images,
            'robot_config': robot_configurations,
            'plane_z': plane_z,
            'return_annotations': return_annotations,
            'detect_on_cropped': detect_on_cropped,
            'max_detections': max_detections
        }

        self.request_queue.put((client_id, self.response_queues[client_id], params))
        return self.response_queues[client_id].get()

    def get_block_positions_depth(self, images, depths, robot_configurations, return_annotations=True,
                                  block_half_size=0.02, detect_on_cropped=True, max_detections=5):
        raise NotImplementedError("Not implemented for shared position estimator")

    def bboxes_cropped_to_orig(self, bboxes, xyxy):
        raise NotImplementedError("Not implemented for shared position estimator")

    def points_image_to_camera_frame(self, points_image_xy, z_depth):
        raise NotImplementedError("Not implemented for shared position estimator")

    def get_z_depth_mean(self, points_color_image_xy, depth_image, windows_sizes):
        raise NotImplementedError("Not implemented for shared position estimator")

    def get_z_offset_for_depth(self, robot_config, block_half_size):
        raise NotImplementedError("Not implemented for shared position estimator")

    def get_bboxes(self, detect_on_cropped, images, robot_configurations, return_annotations, max_detections):
        raise NotImplementedError("Not implemented for shared position estimator")

    def close(self):
        """Cleanup resources"""
        self.request_queue.put((None, None, None))
        self.service_process.join()
        self.request_queue.close()
        for q in self.response_queues.values():
            q.close()