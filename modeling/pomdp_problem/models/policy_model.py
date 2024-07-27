import pomdp_py
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack


def history_to_belief(initial_belief, history):
    # filter to sensing actions with positive sensing, sensing actions with negative sensing
    # and stack attempt actions with success

    sense_positive = []  # points where we sense there is a block
    sense_negative = []  # points where we sensed there isn't block
    stack_positive = []  # points where we stacked form

    for (action, observation) in history:
        if isinstance(action, ActionSense):
            if observation.is_occupied:
                sense_positive.append((observation.x, observation.y))
            else:
                sense_negative.append((observation.x, observation.y))
        elif isinstance(action, ActionAttemptStack):
            if observation.is_object_picked:
                stack_positive.append((action.x, action.y))



class PolicyModel(pomdp_py.PolicyModel):
    def __init__(self,):
        pass

    def get_all_actions(self, state, history):
        # if steps left is 0 no aplicable actions
        # if steps left is 1 only pick up
        pass

    def rollout(self, state, history):
        pass
