import numpy as np
import pandas as pd
import typer
from experiments_lab.experiment_results_data import ExperimentResults
from lab_ur_stack.utils.workspace_utils import workspace_y_lims_default, workspace_x_lims_default
from modeling.belief.block_position_belief import BlocksPositionsBelief
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
    # extract additional data for each row from the resutls file and save in lists
    belief_mus_after_help = []
    belief_sigmas_after_help = []
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

        after_help_mus, after_help_sigmas = res.beliefs[0].get_mus_and_sigmas()

        belief_mus_after_help.append(after_help_mus)
        belief_sigmas_after_help.append(after_help_sigmas)
        accumulated_rewards.append(res.total_reward)
        episode_lengths.append(len(res.rewards))
        n_successful_pickups.append(curr_num_successful_pickups)
        n_pickup_attempts.append(curr_num_pickup_attempts)
        n_detection_from_help.append(curr_num_detection_from_help)

    # add new data to the table:
    experiments_list['belief_mus_after_help'] = belief_mus_after_help
    experiments_list['belief_sigmas_after_help'] = belief_sigmas_after_help
    experiments_list['accumulated_rewards'] = accumulated_rewards
    experiments_list['episode_lengths'] = episode_lengths
    experiments_list['n_successful_pickups'] = n_successful_pickups
    experiments_list['n_pickup_attempts'] = n_pickup_attempts
    experiments_list['n_detection_from_help'] = n_detection_from_help

    # save the table:
    experiments_list.to_csv(output_file, index=False)


def experiments_results_to_empirical_vd_table(experiment_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given experiments results df, generate a df where for each experiment there added a value difference column
    where it's the difference between the accumulated rewards in that experiment and the accumulated rewards
    in the same state and belief but without help. All original columns are preserved.
    """
    value_diff_results = []
    base_df = experiment_results_df[experiment_results_df['help_config_idx_local'] == -1]

    for _, baseline_row in base_df.iterrows():
        belief_idx = baseline_row['belief_idx']
        state_idx = baseline_row['state_idx']
        baseline_reward = baseline_row['accumulated_rewards']

        filtered_rows = experiment_results_df[
            (experiment_results_df['state_idx'] == state_idx) &
            (experiment_results_df['belief_idx'] == belief_idx) &
            (experiment_results_df['help_config_idx_local'] != -1)
            ]

        for _, row in filtered_rows.iterrows():
            # Create a new dictionary with all columns from the original row
            result_dict = row.to_dict()
            # Add the value difference
            result_dict['value_diff'] = row['accumulated_rewards'] - baseline_reward
            result_dict['baseline_value'] = baseline_reward
            value_diff_results.append(result_dict)

    return pd.DataFrame(value_diff_results)


def add_state_likelihood_column(vd_table: pd.DataFrame) -> pd.DataFrame:
    """
    Given any dataframe with a belief columns and state columns, add a column with the likelihood of each
     state in the belief. perform the operation in place.
    """
    # Get unique belief-state combinations
    unique_pairs = vd_table.groupby(['belief_idx', 'state_idx']).first()

    # Calculate probabilities for unique pairs
    probability_map = {}
    for (belief_idx, state_idx), row in unique_pairs.iterrows():
        belief_mu = eval(row['belief_mus'])
        belief_sigma = eval(row['belief_sigmas'])
        belief = BlocksPositionsBelief(
            len(belief_mu),
            workspace_x_lims_default,
            workspace_y_lims_default,
            init_mus=belief_mu,
            init_sigmas=belief_sigma
        )
        state = eval(row['state'])
        probability_map[(belief_idx, state_idx)] = belief.state_pdf(state)

    # Map probabilities to all rows using the precomputed values
    vd_table['state_likelihood'] = vd_table.apply(
        lambda row: probability_map[(row['belief_idx'], row['state_idx'])],
        axis=1
    )

    return vd_table


def vd_table_to_empirical_voa_table(vd_table: pd.DataFrame) -> pd.DataFrame:
    """
    Given a value difference table, generate a table where each row corresponds to a belief-helpconfig pair
    and there's a column for empirical VOA which is the expected value diff over all states
    (weighted by the likelihood of each state)
    """
    # check that the state probabilities are already calculated
    if 'state_likelihood' not in vd_table.columns:
        vd_table = add_state_likelihood_column(vd_table)

    voa_results = []
    for (belief_idx, help_config_idx), group in vd_table.groupby(['belief_idx', 'help_config_idx_local']):
        # compute the empirical voa
        weights = group['state_likelihood']
        weights /= weights.sum()
        value_diffs = group['value_diff']
        base_values = group['baseline_value']
        empirical_voa = np.sum(weights * value_diffs)[0]
        empirical_baseline = np.sum(weights * base_values)[0]

        # compute empirical population variance
        empirical_population_variance = np.sum(weights * (value_diffs - empirical_voa) ** 2)[0]


        voa_results.append({
            'belief_idx': belief_idx,
            'help_config_idx_local': help_config_idx,
            'belief_mus': group['belief_mus'].iloc[0],
            'belief_sigmas': group['belief_sigmas'].iloc[0],
            'help_config': group['help_config'].iloc[0],
            'empirical_voa': empirical_voa,
            'empirical_voa_population_variance': empirical_population_variance,
            'empirical_baseline_value': empirical_baseline
        })

    return pd.DataFrame(voa_results)


@app.command()
def main():
    pass


if __name__ == "__main__":
    app()

