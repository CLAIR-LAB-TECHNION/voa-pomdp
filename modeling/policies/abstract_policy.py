from abc import abstractmethod

from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *
from modeling.belief.block_position_belief import BlocksPositionsBelief


class AbastractPolicy:
    def __init__(self, ):
        pass

    @abstractmethod
    def sample_action(self,
                      positions_belief: BlocksPositionsBelief,
                      history: list[tuple[ActionBase, ObservationBase]]) -> ActionBase:
        raise NotImplementedError

    def __call__(self, belief: BlocksPositionsBelief, history: list = None) -> ActionBase:
        return self.sample_action(belief, history)
