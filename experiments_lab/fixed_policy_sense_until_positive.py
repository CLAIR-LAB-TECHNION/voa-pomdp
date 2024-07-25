from modeling.belief.block_position_belief import BlocksPositionsBelief



class FixedSenseUntilPositivePolicy:
    def __init__(self, ):
        self.prev_action = None

    def policy(self, positions_belief: BlocksPositionsBelief, last_observation):
        """
        This is a fixed policy that tries to sense the block until it's positive then stack it up.
        At the last step it will try to stack at maximum likelihood position.
        """
        # if this is the last step, or last step we sensed positive, attempt pick up:
        if last_observation[1] == 0 or \
                (self.prev_action[0] == "sense" and last_observation[0] == True):
            pick_up_position = self.max_likelihood_position(positions_belief)
            return "attempt_stack", pick_up_position[0], pick_up_position[1]

        # otherwise, sense for the first block in sampled place from it's belief
        first_block_belief = positions_belief.block_beliefs[0]
        sense_point = first_block_belief.sample()[0]
        return "sense", sense_point[0], sense_point[1]

    def max_likelihood_position(self, positions_belief):
        pass
