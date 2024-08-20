import cv2
import numpy as np
import matplotlib.pyplot as plt
from modeling.belief.belief_plotting import plot_all_blocks_beliefs


class ExperimentVisualizer:
    def __init__(self, window_name="Experiment Visualization", window_size=(1200, 800)):
        self.window_name = window_name
        self.window_size = window_size
        self.canvas = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

        # Create initial blank subplots
        self.experiment_type = ""
        self.accumulated_reward = 0
        self.additional_info = ""
        self.action_obs_reward_text = ""
        self.detection_image = np.zeros((400, 400, 3), dtype=np.uint8)
        self.detection_header = "Last Detections"
        self.belief_image = np.zeros((800, 600, 3), dtype=np.uint8)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])

    def update_experiment_type(self, experiment_type):
        self.experiment_type = experiment_type

    def update_accumulated_reward(self, accumulated_reward):
        self.accumulated_reward = accumulated_reward

    def update_additional_info(self, additional_info):
        self.additional_info = additional_info

    def update_action_obs_reward(self, actions, observations, rewards, k=5):
        text = "Actions | Observations | Rewards\n"
        text += "---------------------------------\n"
        for a, o, r in list(zip(actions, observations, rewards))[-k:]:
            text += f"{str(a)[:15]:15} | {str(o)[:15]:15} | {r:.2f}\n"
        self.action_obs_reward_text = text

    def update_detection_image(self, image, header=None):
        self.detection_image = cv2.resize(image, (400, 400))
        if header is not None:
            self.detection_header = header

    def update_belief(self, belief, positive_sensing_points=None, negative_sensing_points=None,
                      pickup_attempt_points=None):
        fig = plt.figure(figsize=(6, 8))
        plot_all_blocks_beliefs(belief,
                                positive_sensing_points=positive_sensing_points,
                                negative_sensing_points=negative_sensing_points,
                                pickup_attempt_points=pickup_attempt_points)
        fig.canvas.draw()
        belief_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        belief_image = belief_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        self.belief_image = cv2.resize(belief_image, (600, 800))

    def _draw_text(self, image, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
        for i, line in enumerate(text.split('\n')):
            cv2.putText(image, line, (position[0], position[1] + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def update_display(self):
        self.canvas.fill(0)  # Clear the canvas

        # Draw info text
        info_text = f"Experiment Type: {self.experiment_type}\n"
        info_text += f"Accumulated Reward: {self.accumulated_reward:.2f}\n"
        info_text += self.additional_info
        self._draw_text(self.canvas, info_text, (10, 30))

        # Draw action, observation, reward text
        self._draw_text(self.canvas, self.action_obs_reward_text, (10, 150))

        # Draw detection image
        self._draw_text(self.canvas, self.detection_header, (10, 400))
        self.canvas[420:820, 10:410] = self.detection_image

        # Draw belief image
        self.canvas[:800, 600:] = self.belief_image

        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()