import pomdp_py
from modeling.pomdp_problem.agent.agent import Agent


class Environment(pomdp_py.Environment):
    def __init__(self, init_state, transition_model, reward_model):
        super().__init__(init_state=init_state,
                         transition_model=transition_model,
                         reward_model=reward_model,)

    @classmethod
    def from_agent(cls, agent, init_state):
        return cls(init_state=init_state,
                   transition_model=agent.transition_model,
                   reward_model=agent.reward_model)