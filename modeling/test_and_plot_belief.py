from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from modeling.block_position_belief import BlocksPositionsBelief
from modeling.belief_plotting import plot_block_belief, plot_all_blocks_beliefs


def one_block_point_sensing():
    mus = [-0.7, -0.7]
    sigmas = [0.5, 0.2]

    empty_sensing_points = [[-0.8, -0.8], [-0.78, -0.75], [-0.73, -0.73]]

    belief = BlocksPositionsBelief(1, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

    for e_sensing_point in empty_sensing_points:
        belief.update_belief_from_point_sensing(e_sensing_point[0], e_sensing_point[1], is_occupied=False)

    plot_block_belief(belief, 0, negative_sensing_points=empty_sensing_points, grid_size=1000)

    positive_sensing_point = [-0.65, -0.65]
    all_positive_points_so_far = [positive_sensing_point]
    belief.update_belief_from_point_sensing(positive_sensing_point[0], positive_sensing_point[1], is_occupied=True)
    plot_block_belief(belief, 0, negative_sensing_points=empty_sensing_points,
                      positive_sensing_points=all_positive_points_so_far, grid_size=1000)

    positive_sensing_point = [-0.62, -0.62]
    all_positive_points_so_far.append(positive_sensing_point)
    belief.update_belief_from_point_sensing(positive_sensing_point[0], positive_sensing_point[1], is_occupied=True)
    plot_block_belief(belief, 0, negative_sensing_points=empty_sensing_points,
                      positive_sensing_points=all_positive_points_so_far, grid_size=1000)

def multiple_blocks():
    mus = [[-0.7, -0.7], [-0.8, -0.9]]
    sigmas = [[0.5, 0.2], [0.2, 0.5]]

    belief = BlocksPositionsBelief(2, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
    plot_all_blocks_beliefs(belief, grid_size=1000)



if __name__ == "__main__":
    # one_block_point_sensing()
    multiple_blocks()
