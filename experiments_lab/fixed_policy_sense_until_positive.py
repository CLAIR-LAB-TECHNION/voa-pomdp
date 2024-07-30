import numpy as np

from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief



class FixedSenseUntilPositivePolicy:
    def __init__(self, ):
        self.prev_action = None

    def policy(self, positions_belief: UnnormalizedBlocksPositionsBelief, prev_observation):
        """
        This is a fixed policy that tries to sense the block until it's positive then stack it up.
        At the last step it will try to stack at maximum likelihood position.
        """
        if self.prev_action is not None:
            # if this is the last step, or in previous step we sensed positive, attempt pick up:
            if prev_observation[1] == 1 or \
                    (self.prev_action[0] == "sense" and prev_observation[0] == True):
                pick_up_position = self.max_likelihood_position(positions_belief)
                action = ("attempt_stack", pick_up_position[0], pick_up_position[1])
                self.prev_action = action
                return action

        # otherwise, sense for the first block in sampled place from belief
        first_block_belief = positions_belief.block_beliefs[0]
        sense_point = first_block_belief.sample()[0]
        action = ("sense", sense_point[0], sense_point[1])
        self.prev_action = action
        return action

    def max_likelihood_position(self, positions_belief, grid_size=200):
        # compute pdf on a grid:
        x = np.linspace(positions_belief.ws_x_lims[0], positions_belief.ws_x_lims[1], grid_size)
        y = np.linspace(positions_belief.ws_y_lims[0], positions_belief.ws_y_lims[1], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        pdf = np.asarray(positions_belief.block_beliefs[0].pdf(points)).reshape(grid_size, grid_size)
        max_likelihood_idx = np.argmax(pdf)
        max_likelihood_position = points[max_likelihood_idx]
        return max_likelihood_position

    def __call__(self, positions_belief: UnnormalizedBlocksPositionsBelief, prev_observation):
        return self.policy(positions_belief, prev_observation)
