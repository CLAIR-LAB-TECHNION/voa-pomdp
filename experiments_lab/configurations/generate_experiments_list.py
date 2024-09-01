import typer
import pandas as pd
import numpy as np
import os
from datetime import datetime

from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default, \
    sample_block_positions_from_dists
from modeling.belief.block_position_belief import BlocksPositionsBelief, BlockPosDist

app = typer.Typer()


def sample_belief(n_blocks, min_std, max_std, ws_x_lims, ws_y_lims, min_distance_between_mus=0.1):
    block_pos_mu = []
    for _ in range(n_blocks):
        while True:
            mu = np.random.uniform(low=[ws_x_lims[0], ws_y_lims[0]], high=[ws_x_lims[1], ws_y_lims[1]])
            if all(np.linalg.norm(mu - other_mu) >= min_distance_between_mus for other_mu in block_pos_mu):
                block_pos_mu.append(mu)
                break
    block_pos_mu = np.array(block_pos_mu)
    block_pos_sigma = np.random.uniform(low=min_std, high=max_std, size=(n_blocks, 2))
    return BlocksPositionsBelief(n_blocks, ws_x_lims, ws_y_lims, init_mus=block_pos_mu, init_sigmas=block_pos_sigma)


def sample_state(belief: BlocksPositionsBelief, min_dist=0.1):
    return sample_block_positions_from_dists(belief.block_beliefs, min_dist=min_dist)


def load_help_configs(file_path="help_configs.npy"):
    return np.load(file_path)


@app.command(context_settings={"ignore_unknown_options": True})
def generate_experiments(
        n_blocks: int = typer.Option(4, help="Number of blocks in the experiment"),
        num_beliefs: int = typer.Option(2, help="Number of beliefs to sample"),
        min_std: float = typer.Option(0.02, help="Minimum standard deviation for belief sampling"),
        max_std: float = typer.Option(0.15, help="Maximum standard deviation for belief sampling"),
        n_states_per_belief: int = typer.Option(5, help="Number of states to sample per belief"),
        n_helper_configs: int = typer.Option(10, help="Number of helper configurations to sample"),
        output_file: str = typer.Option("experiments", help="Output file name (without extension)"),
):
    # Use the default workspace limits from your utility module
    ws_x_lims = workspace_x_lims_default
    ws_y_lims = workspace_y_lims_default

    # Load help configs
    all_help_configs = load_help_configs()

    # Sample helper configurations once
    help_configs = all_help_configs[np.random.choice(len(all_help_configs), n_helper_configs, replace=False)]

    # Create empty list to store experiment data
    experiments = []

    # Generate experiments
    experiment_id = 1
    for belief_idx in range(num_beliefs):
        belief = sample_belief(n_blocks, min_std, max_std, ws_x_lims, ws_y_lims)

        for state_idx in range(n_states_per_belief):
            state = sample_state(belief)

            # Add experiment without help
            experiments.append({
                'experiment_id': experiment_id,
                'belief_idx': belief_idx,
                'state_idx': state_idx,
                'help_config_idx_local': -1,  # -1 indicates no help
                'belief_mus': [[b.mu_x, b.mu_y] for b in belief.block_beliefs],
                'belief_sigmas': [[b.sigma_x, b.sigma_y] for b in belief.block_beliefs],
                'state': state,
                'help_config': None,
                'conducted_datetime_stamp': ''
            })
            experiment_id += 1

            # Add experiments with help
            for help_config_idx, help_config in enumerate(help_configs):
                experiments.append({
                    'experiment_id': experiment_id,
                    'belief_idx': belief_idx,
                    'state_idx': state_idx,
                    'help_config_idx_local': help_config_idx,
                    'belief_mus': [[b.mu_x, b.mu_y] for b in belief.block_beliefs],
                    'belief_sigmas': [[b.sigma_x, b.sigma_y] for b in belief.block_beliefs],
                    'state': state,
                    'help_config': help_config.tolist(),
                    'conducted_datetime_stamp': ''
                })
                experiment_id += 1

    # Create DataFrame
    df = pd.DataFrame(experiments)

    # Save to CSV
    output_file_with_extension = f"{output_file}_{n_blocks}_blocks.csv"
    if os.path.exists(output_file_with_extension):
        raise FileExistsError(f"File {output_file_with_extension} already exists. Please choose a different name.")

    df.to_csv(output_file_with_extension, index=False)
    print(f"Experiments list saved to {output_file_with_extension}")

    # Save helper configurations to a separate file
    help_configs_file = f"{output_file}_{n_blocks}_blocks_help_configs.npy"
    np.save(help_configs_file, help_configs)
    print(f"Helper configurations saved to {help_configs_file}")


if __name__ == "__main__":
    app()
