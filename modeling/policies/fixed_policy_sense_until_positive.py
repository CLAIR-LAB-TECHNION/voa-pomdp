import numpy as np
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy


class FixedSenseUntilPositivePolicy(AbastractPolicy):
    def __init__(self, ):
        super().__init__()

    def sample_action(self,
                      positions_belief: BlocksPositionsBelief,
                      history: list[tuple[ActionBase, ObservationBase]]):
        """
        This is a fixed policy that tries to sense the block until it's positive then stack it up.
        At the last step it will try to stack at maximum likelihood position.
        """
        prev_action = history[-1][0] if history else None
        prev_observation = history[-1][1] if history else None

        if prev_action is not None:
            # if this is the last step, or in previous step we sensed positive, attempt pick up:
            if prev_observation.steps_left == 1 or \
                    (isinstance(prev_observation, ObservationSenseResult) and prev_observation.is_occupied == True):
                pick_up_position = self.max_likelihood_position(positions_belief)
                action = ActionAttemptStack(pick_up_position[0], pick_up_position[1])
                return action

        # otherwise, sense for the first block in sampled place from belief
        first_block_belief = positions_belief.block_beliefs[0]
        sense_point = first_block_belief.sample()[0]
        action = ActionSense(sense_point[0], sense_point[1])
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

    def get_params(self) -> dict:
        return {}

    def reset(self):
        pass

