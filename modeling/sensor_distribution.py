import numpy as np


def compute_position_stds(detected_positions, camera_position, minimal_std=0.005,
                          distance_coeff=0.02, n_blocks_coeff=0.002,
                          inverse_nearest_block_coeff=0.0005):
    """
    Computes standard deviations for block position distributions based on various factors:
    - minimal_std: base standard deviation (0.5 cm)
    - distance_coeff: increase based on camera distance (2 cm per 1m)
    - n_blocks_coeff: increase based on total number of blocks (0.2 cm per block)
    - inverse_nearest_block_coeff: increase based on nearest block distance (0.5 cm at 10 cm)

    Args:
        detected_positions: numpy array of shape (n_blocks, 2) or (n_blocks, 3)
        camera_position: position of the camera (x, y)
        minimal_std: base standard deviation
        distance_coeff: coefficient for camera distance impact
        n_blocks_coeff: coefficient for number of blocks impact
        inverse_nearest_block_coeff: coefficient for nearest block distance impact

    Returns:
        numpy array of shape (n_blocks, 2) containing std_x and std_y for each block
    """
    n_blocks = len(detected_positions)
    stds = np.full((n_blocks, 2), minimal_std)

    # Add std based on number of blocks
    stds += n_blocks_coeff * (n_blocks - 1)

    detected_positions = np.asarray(detected_positions)
    camera_position = np.asarray(camera_position)
    # Add std based on nearest block distance
    if n_blocks > 1:
        blocks_distances_matrix = np.linalg.norm(detected_positions[:, None] - detected_positions[None, :], axis=-1)
        np.fill_diagonal(blocks_distances_matrix, np.inf)
        nearest_block_distances = np.min(blocks_distances_matrix, axis=1)
        stds += inverse_nearest_block_coeff * nearest_block_distances[:, None]

    # Add std based on camera distance
    camera_distances = np.linalg.norm(detected_positions - camera_position, axis=-1)
    stds += distance_coeff * camera_distances[:, None]

    return stds


def detections_to_distributions(detected_positions,
                                camera_position,
                                minimal_std=0.005,
                                distance_coeff=0.02,
                                n_blocks_coeff=0.002,
                                inverse_nearest_block_coeff=0.0005):
    """
    Converts detected block positions to block position distributions
    the distribution is Gaussian with expectation at detected positions and
    variance that is based on minimal_std and is increased based on the distance
    of the camera from detections (distance_coeff), total num of blocks (n_blocks_coeff)
    and distance of nearest block (inverse_nearest_block_coeff)
    returns expectations [(mu_x, mu_y), ...] and stds [(std_x, std_y), ...]
    """
    if len(detected_positions) == 0:
        return [], []

    expectations = np.asarray(detected_positions)
    # expectations are only x,y:
    expectations = expectations[:, :2]

    stds = compute_position_stds(detected_positions, camera_position,
                                 minimal_std, distance_coeff,
                                 n_blocks_coeff, inverse_nearest_block_coeff)

    return expectations, stds


# test:
if __name__ == "__main__":
    from modeling.belief.block_position_belief import BlocksPositionsBelief
    from modeling.belief.belief_plotting import plot_all_blocks_beliefs
    from lab_ur_stack.utils.workspace_utils import (workspace_x_lims_default,
                                                    workspace_y_lims_default, )

    detected_positions = np.array([[-0.75, -0.75], [-0.6, -0.9]])
    camera_position = np.array([-0.3360082029097331, -0.4844767882805121, 0.7654477924813987])
    expectations, stds = detections_to_distributions(detected_positions, camera_position)

    distributions = BlocksPositionsBelief(2, workspace_x_lims_default, workspace_y_lims_default,
                                          init_mus=expectations, init_sigmas=stds)
    plot_all_blocks_beliefs(distributions)
    print(stds)

    #######

    detected_positions = np.array([[-0.75, -0.75], [-0.6, -0.9], [-0.6, -0.6]])
    expectations, stds = detections_to_distributions(detected_positions, camera_position)
    distributions = BlocksPositionsBelief(3, workspace_x_lims_default, workspace_y_lims_default,
                                          init_mus=expectations, init_sigmas=stds)
    plot_all_blocks_beliefs(distributions)
    print(stds)

    #######
    camera_position = np.array([-0.461, -0.64, 0.50])
    expectations, stds = detections_to_distributions(detected_positions, camera_position)
    distributions = BlocksPositionsBelief(3, workspace_x_lims_default, workspace_y_lims_default,
                                          init_mus=expectations, init_sigmas=stds)
    plot_all_blocks_beliefs(distributions)
    print(stds)