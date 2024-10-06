from copy import deepcopy

from experiments_lab.experiment_results_data import ExperimentResults
from experiments_lab.utils import plot_belief_with_history, update_belief
from experiments_sim.block_stacking_simulator import BlockStackingSimulator
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.policies.abstract_policy import AbastractPolicy
import logging

from modeling.pomdp_problem.domain.observation import ObservationReachedTerminal


def run_single_experiment(env: BlockStackingSimulator,
                          policy: AbastractPolicy,
                          init_block_positions,
                          init_block_belief: BlocksPositionsBelief,
                          help_config=None,
                          plot_belief=False, ):
    logging.info("EXPR running single experiment")

    results = ExperimentResults(policy_type=policy.__class__.__name__,
                                agent_params=policy.get_params(),
                                help_config=help_config)
    results.actual_initial_block_positions = init_block_positions
    results.belief_before_help = deepcopy(init_block_belief)
    actual_states = [env.mujoco_env.get_block_positions()]
    results.set_metadata("actual_states", actual_states)

    env.reset(init_block_positions)

    if help_config is not None:
        # TODO copy from lab exprmgr (take to utils and import)
        results.is_with_helper = True

        current_belief = None
        raise NotImplementedError()
    else:
        results.is_with_helper = False
        current_belief = deepcopy(init_block_belief)

    results.beliefs.append(current_belief)

    policy.reset(current_belief)
    history = []
    accumulated_reward = 0
    for i in range(env.max_steps):
        if plot_belief:
            plot_belief_with_history(current_belief, actual_state=results.get_metadata("actual_states")[-1],
                                     history=history)

        action = policy.sample_action(current_belief, history)
        observation, reward = env.step(action)

        accumulated_reward += reward
        history.append((action, observation))

        current_belief = update_belief(current_belief, action, observation)

        results.beliefs.append(current_belief)
        results.actions.append(action)
        results.observations.append(observation)
        results.rewards.append(reward)
        actual_states.append(env.mujoco_env.get_block_positions())

        logging.info(f"EXPR step {i}. reward: {reward}, action: {action},  observation: {observation}, "
                     f"actual state: {actual_states[-1]}, accumulated reward: {accumulated_reward}")

        if isinstance(observation, ObservationReachedTerminal) or len(current_belief.block_beliefs) == 0 \
                or observation.steps_left <= 0:
            break

    results.total_reward = accumulated_reward

    logging.info(f"EXPR Experiment finished with total reward: {accumulated_reward}."
                 f" picked up {env.n_picked_blocks} blocks")

    return results
