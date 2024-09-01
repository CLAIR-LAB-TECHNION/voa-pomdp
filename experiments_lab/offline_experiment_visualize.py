import os
import typer
import cv2
import numpy as np
from experiments_lab.experiment_results_data import ExperimentResults
from experiment_visualizer import ExperimentVisualizer
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.belief.belief_plotting import plot_all_blocks_beliefs

app = typer.Typer()


def plot_belief(current_belief, actual_state, history=[]):
    # TODO: add actual block positions
    # TODO other details from Experiment results
    positive_sens = []
    negative_sens = []
    failed_pickups = []
    for a, o in history:
        if isinstance(o, ObservationSenseResult):
            if o.is_occupied:
                positive_sens.append((a.x, a.y))
            else:
                negative_sens.append((a.x, a.y))
        elif isinstance(o, ObservationStackAttemptResult):
            if not o.is_object_picked:
                failed_pickups.append((a.x, a.y))

    return plot_all_blocks_beliefs(current_belief,
                                   actual_states=actual_state,
                                   positive_sensing_points=positive_sens,
                                   negative_sensing_points=negative_sens,
                                   pickup_attempt_points=failed_pickups,
                                   ret_as_image=True)


@app.command()
def visualize_experiment(experiment_dir: str):
    file_name = "results.pkl"
    file_path = os.path.join(experiment_dir, file_name)

    results = ExperimentResults.load(file_path)

    visualizer = ExperimentVisualizer()
    visualizer.start()

    try:
        i = 0
        while i < len(results.beliefs):
            visualizer.update_experiment_type(
                f"Step {i + 1}/{len(results.beliefs)}")
            visualizer.update_accumulated_reward(sum(results.rewards[:i + 1]))

            if i > 0:
                visualizer.update_action_obs_reward(
                    results.actions[:i],
                    results.observations[:i],
                    results.rewards[:i]
                )

            history = list(zip(results.actions[:i], results.observations[:i]))
            belief_image = plot_belief(results.beliefs[i],
                                       actual_state=results.actual_initial_block_positions,
                                       history=history)
            visualizer.update_belief_image(belief_image)

            # if i == 0 and with_help:
            #     detection_image_path = os.path.join(os.path.dirname(results._file_path), "help_detections.png")
            #     if os.path.exists(detection_image_path):
            #         detection_image = cv2.imread(detection_image_path)
            #         visualizer.update_detection_image(detection_image, "Help Detections")

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
