import numpy as np
import pandas as pd
import typer
from experiments_lab.experiment_results_data import ExperimentResults
from modeling.pomdp_problem.domain.observation import ObservationStackAttemptResult

app = typer.Typer()

@app.command()
def process_experiment_results_to_table(
        experiment_list_file=typer.Option(help="Path to the file containing the list of experiments to process", prompt=True),
        experiment_results_dir=typer.Option(help="Path to the directory containing the results of the experiments", prompt=True),
        output_file=typer.Option(help="Path to the output file", prompt=True)):
    """
    extract data from experiments results and save in a copy of the experiment list table
    python -m data_processing.process_experiments_results process-experiment-results-to-table\
    --experiment-list-file="experiments_sim/configurations/experiments_planner_2000.csv"\
    --experiment-results-dir="experiments_sim/experiments"\
    --output-file="experiments_sim/results/experiments_planner_2000_results.csv"
    """
    # load experiments list:
    experiments_list = pd.read_csv(experiment_list_file)

    print(f"Processing {len(experiments_list)} experiments...")
    # extract additional data for each row from the resutls fil
    #
    # e and save in lists
    accumulated_rewards = []
    episode_lengths = []
    n_successful_pickups = []
    n_pickup_attempts = []
    n_detection_from_help = []
    for index, row in experiments_list.iterrows():
        conducted_datetime_stamp = row['conducted_datetime_stamp']
        # Define the path to the folder and the .pkl file
        pkl_file_path = f"{experiment_results_dir}/{conducted_datetime_stamp}/results.pkl"
        res = ExperimentResults.load(pkl_file_path)

        if res is None:
            print(f"Warning: No results found for experiment {row['experiment_id']}, {conducted_datetime_stamp}."
                  f" skipping...")
            continue

        curr_num_pickup_attempts = sum([1 for obs in res.observations
                                        if isinstance(obs, ObservationStackAttemptResult)])
        curr_num_successful_pickups = sum([1 for obs in res.observations
                                           if isinstance(obs, ObservationStackAttemptResult) and obs.is_object_picked])
        if res.help_detections_sigmas is None:
            curr_num_detection_from_help = 0
        else:
            # sigmas that are lower than zeros means no detections
            curr_num_detection_from_help = sum([sigma[0] >= 0 for sigma in res.help_detections_sigmas])

        accumulated_rewards.append(res.total_reward)
        episode_lengths.append(len(res.rewards))
        n_successful_pickups.append(curr_num_successful_pickups)
        n_pickup_attempts.append(curr_num_pickup_attempts)
        n_detection_from_help.append(curr_num_detection_from_help)

    # add new data to the table:
    experiments_list['accumulated_rewards'] = accumulated_rewards
    experiments_list['episode_lengths'] = episode_lengths
    experiments_list['n_successful_pickups'] = n_successful_pickups
    experiments_list['n_pickup_attempts'] = n_pickup_attempts
    experiments_list['n_detection_from_help'] = n_detection_from_help

    # save the table:
    experiments_list.to_csv(output_file, index=False)


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()

