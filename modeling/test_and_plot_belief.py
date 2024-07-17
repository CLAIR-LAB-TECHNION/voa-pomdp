from lab_ur_stack.utils.workspace_utils import workspace_x_lims_default, workspace_y_lims_default
from modeling.block_position_belief import BlocksPositionsBelief
from modeling.belief_plotting import plot_block_belief
#
mus = [-0.7, -0.7]
sigmas = [0.5, 0.2]

belief = BlocksPositionsBelief(1, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
belief.add_empty_area([-0.85, -0.77], [-0.73, -0.65])
belief.add_empty_area([-0.89, -0.81], [-0.75, -0.67])
belief.add_empty_area([-0.8, -0.72], [-0.72, -0.64])

plot_block_belief(belief, 0, grid_size=1000)
