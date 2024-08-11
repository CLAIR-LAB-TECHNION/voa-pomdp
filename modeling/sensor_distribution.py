import numpy as np


def detections_to_distributions(detected_positions,
                                camera_position,
                                minimal_std=0.005,  # 0.5 cm
                                distance_coeff=0.01,  # 1 cm added for distance of 1m
                                n_blocks_coeff=0.001,  # each block adds 0.1 cm
                                inverse_nearest_block_coeff=0.0005  # block at 10 cm adds 0.5 cm
                                ):
    """
    Converts detected block positions to block position distributions
    the distribution is Gaussian with expectation at detected positions and
    variance that is based on minimal_std and is increased based on the distance
    of the camera from detections (distance_coeff), total num of blocks (n_blocks_coeff)
    and distance of nearest block (inverse_nearest_block_coeff)

    returns expectations [(mu_x, mu_y), ...] and stds [(std_x, std_y), ...]
    """
    expectations = detected_positions

    n_blocks = len(detected_positions)
    stds = np.full((n_blocks, 2), minimal_std)
    stds += n_blocks_coeff * n_blocks

    blocks_distances_matrix = np.linalg.norm(detected_positions[:, None] - detected_positions[None, :], axis=-1)
    np.fill_diagonal(blocks_distances_matrix, np.inf)
    nearest_block_distances = np.min(blocks_distances_matrix, axis=1)
    stds += inverse_nearest_block_coeff * nearest_block_distances[:, None]

    camera_distances = np.linalg.norm(detected_positions - camera_position, axis=-1)
    stds += distance_coeff * camera_distances[:, None]

    return expectations, stds
