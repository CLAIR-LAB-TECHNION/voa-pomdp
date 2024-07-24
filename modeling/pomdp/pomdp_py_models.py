from copy import deepcopy
import pomdp_py
import numpy as np
from modeling.pomdp.pomdp_py_domain import State
from modeling.pomdp.pomdp_py_domain import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp.pomdp_py_domain import ActionAttemptStack, ActionSense


class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, block_size=0.04, successful_grasp_offset_x=0.01, successful_grasp_offset_y=0.01):
        self.block_size = block_size
        self.successful_grasp_offset_x = successful_grasp_offset_x
        self.successful_grasp_offset_y = successful_grasp_offset_y

    def sample(self, state, action):
        if isinstance(action, ActionAttemptStack):
            return self.sample_stack_attempt(state, action)
        elif isinstance(action, ActionSense):
            return self.sample_sense(state, action)

    def sample_stack_attempt(self, state, action):
        next_state = deepcopy(state)
        next_state.steps_left -= 1

        pick_pos = np.array([action.x, action.y])
        nearest_block = np.argmin(np.linalg.norm(state.block_positions - pick_pos, axis=1))
        nearest_block_pos = state.block_positions[nearest_block]

        # we assume deterministic here
        success = np.abs(nearest_block_pos[0] - pick_pos[0]) < self.successful_grasp_offset_x and \
                  np.abs(nearest_block_pos[1] - pick_pos[1]) < self.successful_grasp_offset_y

        if success:
            next_state.block_positions = np.delete(state.block_positions, nearest_block, axis=0)

        next_state.last_stack_attempt_succeded = success

        return next_state

    def sample_sense(self, state, action):
        # sensing doesn't change the state, it only takes time and provides observation
        next_state = deepcopy(state)
        next_state.steps_left -= 1
        next_state.last_stack_attempt_succeded = None
        return next_state

    def probability(self, next_state, state, action):
        raise NotImplementedError  # don't need this for sampling based methods?

    def argmax(self, state, action):
        raise NotImplementedError



class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, block_size=0.04):
        self.block_size = block_size

    def sample(self, next_state, action):
        if isinstance(action, ActionAttemptStack):
            # for simplicity, right now we assume that the robot can accurately sense if block is picked
            # We can make this probabilistic if we see this is not accurate enough
            return ObservationStackAttemptResult(next_state.last_stack_attempt_succeded, next_state.steps_left)
        elif isinstance(action, ActionSense):
            return self.sample_sense(next_state, action)

    def sample_sense(self, next_state, action):
        # check if there is block within block_size distance from the point both in x and y
        # if there is, it is occupied
        for block_pos in next_state.block_positions:
            if np.abs(block_pos[0] - action.x) < self.block_size and np.abs(block_pos[1] - action.y) < self.block_size:
                return ObservationSenseResult(action.x, action.y, True, next_state.steps_left)

        return ObservationSenseResult(action.x, action.y, False, next_state.steps_left)

    def probability(self, observation, next_state, action):
        raise NotImplementedError  # don't need this for sampling based methods?

    def argmax(self, next_state, action):
        raise NotImplementedError



# class PolicyModel(pomdp_py.PolicyModel)


"""
Note that the sample function is not used directly during planning with POMCP or POUCT;
 Instead, the rollout policy’s sampling process is defined through the rollout function;
  In the example here, indeed, you could explicitly say that the rollout sampling is just sampling
   from this policy model through the sample function.
"""
