import time
from copy import deepcopy

import numpy as np

from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.belief.belief_plotting import plot_block_belief, plot_all_blocks_beliefs


def one_block_point_sensing():
    mus = [-0.7, -0.7]
    sigmas = [0.5, 0.2]

    empty_sensing_points = [[-0.8, -0.8], [-0.78, -0.75], [-0.73, -0.73]]

    belief = BlocksPositionsBelief(1, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

    for e_sensing_point in empty_sensing_points:
        belief.update_from_point_sensing_observation(e_sensing_point[0], e_sensing_point[1], is_occupied=False)

    plot_block_belief(belief, 0, negative_sensing_points=empty_sensing_points, grid_size=1000)

    positive_sensing_point = [-0.65, -0.65]
    all_positive_points_so_far = [positive_sensing_point]
    belief.update_from_point_sensing_observation(positive_sensing_point[0], positive_sensing_point[1], is_occupied=True)
    plot_block_belief(belief, 0, negative_sensing_points=empty_sensing_points,
                      positive_sensing_points=all_positive_points_so_far, grid_size=1000)

    positive_sensing_point = [-0.62, -0.62]
    all_positive_points_so_far.append(positive_sensing_point)
    belief.update_from_point_sensing_observation(positive_sensing_point[0], positive_sensing_point[1], is_occupied=True)
    plot_block_belief(belief, 0, negative_sensing_points=empty_sensing_points,
                      positive_sensing_points=all_positive_points_so_far, grid_size=1000)


def multiple_blocks():
    block_positions = sample_block_positions_uniform(5, workspace_x_lims_default, workspace_y_lims_default)

    mus = block_positions
    sigmas = [[0.05, 0.2], [0.25, 0.08], [0.1, 0.15], [0.15, 0.15], [0.02, 0.03]]
    positive_sensing_points = [[-0.65, -0.65]]
    negative_sensing_points = [[-0.7, -0.7], [-0.75, -0.75]]
    pickup_points = [[-0.77, -0.65]]

    init_belief = BlocksPositionsBelief(5, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
    belief = deepcopy(init_belief)

    plot_all_blocks_beliefs(belief, grid_size=200)

    belief.update_from_point_sensing_observation(*negative_sensing_points[0], is_occupied=False)
    belief.update_from_point_sensing_observation(*negative_sensing_points[1], is_occupied=False)
    plot_all_blocks_beliefs(belief, grid_size=200, actual_states=mus)

    belief.update_from_point_sensing_observation(*positive_sensing_points[0], is_occupied=True)
    plot_all_blocks_beliefs(belief, grid_size=200)

    belief.update_from_pickup_attempt(*pickup_points[0], observed_success=True)
    plot_all_blocks_beliefs(belief, grid_size=200)

    # should be equivalent to:
    belief2 = deepcopy(init_belief)
    belief2.update_from_history_of_sensing_and_pick_up(positive_sensing_points,
                                                       negative_sensing_points,
                                                       pickup_points,
                                                       [])
    plot_all_blocks_beliefs(belief2, grid_size=200)



def from_detections():
    mus = [[-0.9, -0.9], [-0.75, -0.75], [-0.65, -0.65]]
    sigmas = [[0.5, 0.2], [0.25, 0.08], [0.1, 0.15]]

    detections = [[-0.73, -0.74], [-0.78, -0.77], [-0.75, -0.75]]
    detection_sigmas = [[0.02, 0.02], ]*3

    belief = BlocksPositionsBelief(3, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
    initial_belief = deepcopy(belief)

    mus_and_sigmas_associations = belief.update_from_image_detections_position_distribution(detections, detection_sigmas)
    plot_all_blocks_beliefs(initial_belief, grid_size=100, per_block_observed_mus_and_sigmas=mus_and_sigmas_associations)

    plot_all_blocks_beliefs(belief, grid_size=200)


if __name__ == "__main__":
    # one_block_point_sensing()
    multiple_blocks()
    # from_detections()

    # block_positions = sample_block_positions_uniform(5, workspace_x_lims_default, workspace_y_lims_default)
    #
    # mus = block_positions
    # sigmas = [[0.05, 0.2], [0.25, 0.08], [0.1, 0.15], [0.15, 0.15], [0.02, 0.03]]
    # positive_sensing_points = [[-0.65, -0.65]]
    # negative_sensing_points = [[-0.7, -0.7], [-0.75, -0.75]]
    # pickup_points = [[-0.77, -0.65]]
    #
    # belief = BlocksPositionsBelief(5, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
    # belief.update_from_history_of_sensing_and_pick_up(positive_sensing_points,
    #                                                    negative_sensing_points,
    #                                                    successful_pickup_points=[])
    #
    # # test the amount of time it will take to compute pdf on a grid:
    # tstart = time.time()
    # grid_size = 100
    # x = np.linspace(*belief.ws_x_lims, grid_size)
    # y = np.linspace(*belief.ws_y_lims, grid_size)
    # xx, yy = np.meshgrid(x, y)
    # points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    # pdf = np.asarray(belief.block_beliefs[0].pdf(points)).reshape(grid_size, grid_size)
    # print(f"pdf computed in {time.time() - tstart} seconds")
    # max_likelihood_idx = np.argmax(pdf)
    # print(f"max likelihood computed in {time.time()-tstart} seconds")
    #
    # plot_block_belief(belief, 0, positive_sensing_points=points)
    #
    # # test the amount of time to compute pdf on 100 samples from it:
    # tstart = time.time()
    # points = belief.block_beliefs[0].sample_with_redundency(200)
    # pdf = np.asarray(belief.block_beliefs[0].pdf(points))
    # print(f"pdf computed in {time.time() - tstart} seconds")
    # max_likelihood_idx = np.argmax(pdf)
    # print(f"max likelihood computed in {time.time()-tstart} seconds")
    # plot_block_belief(belief, 0, positive_sensing_points=points)

