import time
import cv2
import numpy as np
import threading

from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult


class ExperimentVisualizer:
    def __init__(self, window_name="Experiment Visualization", window_size=(1200, 1000)):
        self.window_name = window_name
        self.window_size = window_size
        self.canvas = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

        self.experiment_type = ""
        self.accumulated_reward = 0
        self.additional_info = ""
        self.action_obs_reward_text = ""
        self.detection_image = np.zeros((450, 400, 3), dtype=np.uint8)
        self.detection_header = "Last Detections"
        self.belief_image = np.zeros((1000, 348, 3), dtype=np.uint8)
        self.detections_distributions_image = None

        self.running = False
        self.update_thread = None

    def start(self):
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()

    def stop(self):
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        cv2.destroyAllWindows()

    def reset(self):
        self.experiment_type = ""
        self.accumulated_reward = 0
        self.additional_info = ""
        self.action_obs_reward_text = ""
        self.detection_image = np.zeros((450, 400, 3), dtype=np.uint8)
        self.detection_header = "Last Detections"
        self.belief_image = np.zeros((1000, 348, 3), dtype=np.uint8)
        self.detections_distributions_image = None

    def _update_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

        while self.running:
            self._update_display()
            key = cv2.waitKey(1000) & 0xFF
            if key == 27:
                self.running = False

    def update_experiment_type(self, experiment_type):
        self.experiment_type = experiment_type

    def update_accumulated_reward(self, accumulated_reward):
        self.accumulated_reward = accumulated_reward

    def update_additional_info(self, additional_info):
        self.additional_info = additional_info

    def update_action_obs_reward(self, actions, observations, rewards):
        text = "Actions | Observations | Rewards\n"
        text += "---------------------------------\n"
        for a, o, r in list(zip(actions, observations, rewards))[-12:]:
            next_line = ""
            if isinstance(a, ActionSense):
                next_line += f"SE({a.x:5.4f}, {a.y:5.4f}) |"
            elif isinstance(a, ActionAttemptStack):
                next_line += f"ST({a.x:5.4f}, {a.y:5.4f}) |"

            if isinstance(o, ObservationSenseResult):
                next_line += f" occupied |" if o.is_occupied else f" empty |"
            elif isinstance(o, ObservationStackAttemptResult):
                next_line += f" picked |" if o.is_object_picked else f" not picked |"

            next_line += f" {r:5.4f}\n"

            text += next_line

        self.action_obs_reward_text = text

    def update_detection_image(self, image, header=None):
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_w = min(400, w)
        new_h = int(new_w / aspect_ratio)
        if new_h > 450:
            new_h = 450
            new_w = int(new_h * aspect_ratio)
        new_w, new_h = int(new_w), int(new_h)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.detection_image = np.zeros((450, 400, 3), dtype=np.uint8)
        y_offset = (450 - new_h) // 2
        x_offset = (400 - new_w) // 2
        self.detection_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        if header is not None:
            self.detection_header = header

    def update_belief_image(self, belief_image):
        belief_image = np.asarray(belief_image, dtype=np.uint8)
        if belief_image.ndim == 2:
            belief_image = cv2.cvtColor(belief_image, cv2.COLOR_GRAY2BGR)
        elif belief_image.shape[2] == 4:
            belief_image = cv2.cvtColor(belief_image, cv2.COLOR_RGBA2BGR)

        h, w = belief_image.shape[:2]
        aspect_ratio = w / h
        new_h = min(1000, h)
        new_w = int(new_h * aspect_ratio)
        if new_w > 348:
            new_w = 348
            new_h = int(new_w / aspect_ratio)
        new_w, new_h = int(new_w), int(new_h)
        resized = cv2.resize(belief_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.belief_image = np.zeros((1000, 348, 3), dtype=np.uint8)
        y_offset = (1000 - new_h) // 2
        x_offset = (348 - new_w) // 2
        self.belief_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    def add_detections_distributions(self, image):
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_h = 1000
        new_w = int(new_h * aspect_ratio)
        if new_w > 348:
            new_w = 348
            new_h = int(new_w / aspect_ratio)
        new_w, new_h = int(new_w), int(new_h)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.detections_distributions_image = np.zeros((1000, 348, 3), dtype=np.uint8)
        y_offset = (1000 - new_h) // 2
        x_offset = (348 - new_w) // 2
        self.detections_distributions_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    def _draw_text(self, image, text, position, font_scale=0.7, color=(255, 255, 255), thickness=1):
        for i, line in enumerate(text.split('\n')):
            cv2.putText(image, line, (position[0], position[1] + i * 25),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

    def _update_display(self):
        self.canvas.fill(0)

        info_text = f"{self.experiment_type}\n"
        info_text += f"Accumulated Reward: {self.accumulated_reward:.4f}\n"
        info_text += self.additional_info
        self._draw_text(self.canvas, info_text, (10, 30), font_scale=0.6)

        self._draw_text(self.canvas, self.action_obs_reward_text, (10, 150), font_scale=0.6)

        self._draw_text(self.canvas, self.detection_header, (10, 520))
        self.canvas[550:1000, 10:410] = self.detection_image

        if self.detections_distributions_image is not None:
            self.canvas[:1000, 500:848] = self.belief_image
            self.canvas[:1000, 852:1200] = self.detections_distributions_image
            self._draw_text(self.canvas, "", (500, 20))
            self._draw_text(self.canvas, "", (852, 20))
        else:
            self.canvas[:1000, 500:848] = self.belief_image
            self._draw_text(self.canvas, "Belief", (500, 20))

        cv2.imshow(self.window_name, self.canvas)


if __name__ == "__main__":
    visualizer = ExperimentVisualizer()
    visualizer.start()

    try:
        for i in range(50):
            visualizer.update_experiment_type(f"Test {i}")
            visualizer.update_accumulated_reward(i * 10)
            visualizer.update_additional_info(f"Step {i}")
            visualizer.update_action_obs_reward(
                [ActionSense(0.1 * j, 0.1 * j) if j % 2 == 0 else ActionAttemptStack(0.1 * j, 0.1 * j) for j in
                 range(i - 11, i + 1)],
                [ObservationSenseResult(j % 2 == 0, [0.1 * j, 0.1 * j],
                                        1) if j % 2 == 0 else ObservationStackAttemptResult(j % 3 == 0,
                                                                                            [0.1 * j, 0.1 * j], 1) for j
                 in range(i - 11, i + 1)],
                [j * 0.5 for j in range(i - 11, i + 1)]
            )
            visualizer.update_detection_image(np.random.randint(0, 255, (700, 700, 3)))
            visualizer.update_belief_image(np.random.randint(0, 255, (1000, 600, 3)))

            if i % 5 == 0:
                visualizer.add_detections_distributions(np.random.randint(0, 255, (1000, 600, 3)))

            time.sleep(0.5)
    finally:
        visualizer.stop()