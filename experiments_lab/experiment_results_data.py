import pickle
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.domain.action import *
from modeling.pomdp_problem.domain.observation import *


class ExperimentResults:
    def __init__(self, policy_type: str, agent_params: Dict[str, Any]):
        self.policy_information: Dict[str, Any] = {"type": policy_type, "params": agent_params}

        self.actual_initial_block_positions: List[List[float]] = []
        self.total_reward = 0

        self.beliefs: List[BlocksPositionsBelief] = []
        self.actions: List[Any] = []
        self.observations: List[Any] = []
        self.rewards: List[float] = []


        self._metadata: Dict[str, Any] = {}

    def set_metadata(self, key: str, value: Any):
        """Set metadata for the experiment."""
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata for the experiment."""
        return self._metadata.get(key)

    def save(self, filename: str):
        """Save the experiment results to a file."""
        data = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        data['_metadata'] = self._metadata
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> 'ExperimentResults':
        """Load experiment results from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        results = cls()
        for key, value in data.items():
            if key != '_metadata':
                setattr(results, key, value)
        results._metadata = data['_metadata']
        return results

    def visualize_beliefs(self, start_idx: int = 0, end_idx: Optional[int] = None, step: int = 1):
        """Visualize beliefs."""
        if end_idx is None:
            end_idx = len(self.beliefs)
        for i in range(start_idx, end_idx, step):
            plot_all_blocks_beliefs(self.beliefs[i], grid_size=200)


# Testing :

def create_experiment_results(initial_belief, actions, observations, rewards, actual_initial_block_positions):
    results = ExperimentResults()
    belief = deepcopy(initial_belief)
    results.beliefs.append(belief)
    results.actual_initial_block_positions = actual_initial_block_positions

    for action, observation, reward in zip(actions, observations, rewards):
        belief = updated_belief(belief, action, observation)
        results.beliefs.append(belief)
        results.actions.append(action)
        results.observations.append(observation)
        results.rewards.append(reward)
    return results

def updated_belief(belief, action, observation):
    updated_belief = deepcopy(belief)
    if isinstance(action, ActionSense):
        updated_belief.update_from_point_sensing_observation(action.x, action.y, observation.is_occupied)
    elif isinstance(action, ActionAttemptStack):
        updated_belief.update_from_pickup_attempt(action.x, action.y, observation.is_object_picked)
    return updated_belief


# Example usage
if __name__ == "__main__":
    initial_belief = BlocksPositionsBelief(n_blocks=3,
                                           ws_x_lims=[-1, 1],
                                           ws_y_lims=[-1, 1],
                                           init_mus=[[0, 0], [0.2, 0.2], [-0.2, -0.2]],
                                           init_sigmas=[[0.05, 0.04], [0.15, 0.1], [0.1, 0.15]])

    actual_initial_block_positions = [[-0.8, -0.75], [-0.6, -0.65], [-0.7, -0.7]]
    actions = [ActionSense(-0.7, -0.7), ActionAttemptStack(-0.8, -0.8)]
    observations = [ObservationSenseResult(True, [-0.7, -0.7], 1),
                    ObservationStackAttemptResult(True, (-0.5, -1), 0)]
    rewards = [-0.1, 0.9]

    results = create_experiment_results(initial_belief, actions, observations, rewards, actual_initial_block_positions)

    # Add some metadata
    results.set_metadata('datetime', time.strftime("%Y-%m-%d_%H-%M-%S"))

    results.save('experiment_results.pkl')

    # Load and visualize the results
    loaded_results = ExperimentResults.load('experiment_results.pkl')
    loaded_results.visualize_beliefs()
