import os
import typer
import cv2
from experiments_lab.experiment_results_data import ExperimentResults
from experiment_visualizer import ExperimentVisualizer
from experiments_lab.utils import plot_belief_with_history

app = typer.Typer()


@app.command()
def visualize_experiment(experiment_dir: str):
    file_name = "results.pkl"
    file_path = os.path.join(experiment_dir, file_name)

    results = ExperimentResults.load(file_path)

    visualizer = ExperimentVisualizer()
    visualizer.start()

    try:
        if results.is_with_help:
            detection_image_path = os.path.join(experiment_dir, "help_detections.png")
            if os.path.exists(detection_image_path):
                detection_image = cv2.imread(detection_image_path)
                visualizer.update_detection_image(detection_image, "Help Detections")

            if results.help_detections_mus is not None:
                observed_mus_and_sigmas = [(results.help_detections_mus[i], results.help_detections_sigmas[i])
                                           for i in range(len(results.help_detections_mus))]
                belief_im_with_detections = plot_belief_with_history(results.belief_before_help,
                                                                     actual_state=results.actual_initial_block_positions,
                                                                     history=[],
                                                                     observed_mus_and_sigmas=observed_mus_and_sigmas)
                visualizer.add_detections_distributions(belief_im_with_detections)

        i = 0
        while i < len(results.beliefs):
            visualizer.update_experiment_type(
                f"Step {i}/{len(results.beliefs)}")
            visualizer.update_accumulated_reward(sum(results.rewards[:i + 1]))

            if i > 0:
                visualizer.update_action_obs_reward(
                    results.actions[:i],
                    results.observations[:i],
                    results.rewards[:i]
                )

            history = []
            belief = results.beliefs[i]
            history = list(zip(results.actions[:i], results.observations[:i]))

            belief_image = plot_belief_with_history(belief,
                                                    actual_state=results.actual_initial_block_positions,
                                                    history=history, )
            visualizer.update_belief_image(belief_image)

            print("Press 'n' for next, 'p' for previous, or 'q' to quit.")
            key = input().lower()
            if key == 'n':
                i += 1
            elif key == 'p':
                i = max(0, i - 1)
            elif key == 'q':
                break

    finally:
        visualizer.stop()


if __name__ == "__main__":
    app()
