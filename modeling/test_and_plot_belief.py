from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from lab_ur_stack.utils.workspace_utils import sample_block_positions
from modeling.block_position_belief import BlocksPositionsBelief
from modeling.belief_plotting import plot_block_belief, plot_all_blocks_beliefs


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
    block_positions = sample_block_positions(5, workspace_x_lims_default, workspace_y_lims_default)

    mus = block_positions
    sigmas = [[0.05, 0.2], [0.25, 0.08], [0.1, 0.15], [0.15, 0.15], [0.02, 0.03]]

    belief = BlocksPositionsBelief(5, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
    plot_all_blocks_beliefs(belief, grid_size=200)

    belief.update_from_point_sensing_observation(-0.7, -0.7, is_occupied=False)
    belief.update_from_point_sensing_observation(-0.75, -0.75, is_occupied=False)
    plot_all_blocks_beliefs(belief, grid_size=200, actual_states=mus)

    belief.update_from_point_sensing_observation(-0.65, -0.65, is_occupied=True)
    plot_all_blocks_beliefs(belief, grid_size=200)

def from_detections():
    mus = [[-0.9, -0.9], [-0.75, -0.75], [-0.65, -0.65]]
    sigmas = [[0.5, 0.2], [0.25, 0.08], [0.1, 0.15]]

    belief = BlocksPositionsBelief(3, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
    plot_all_blocks_beliefs(belief, grid_size=200)

    detections = [[-0.73, -0.74], [-0.78, -0.77], [-0.75, -0.75]]
    detection_sigmas = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
    belief.update_from_image_detections_position_distribution(detections, detection_sigmas)
    plot_all_blocks_beliefs(belief, grid_size=200)


if __name__ == "__main__":
    # one_block_point_sensing()
    # multiple_blocks()
    from_detections()
