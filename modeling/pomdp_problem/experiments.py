import pomdp_py
from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform, workspace_x_lims_default, \
    workspace_y_lims_default
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.models.policy_model import BeliefModel
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack
from modeling.belief.belief_plotting import plot_all_blocks_beliefs


def get_positions_and_init_belief():
    block_positions = sample_block_positions_uniform(3, workspace_x_lims_default, workspace_y_lims_default)
    mus = block_positions
    sigmas = [[0.05, 0.2], [0.25, 0.08], [0.1, 0.15], [0.15, 0.15], [0.02, 0.03]]
    sigmas = sigmas[:3]
    belief = BeliefModel(3, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

    return block_positions, belief


max_steps = 10
stacking_reward = 1
cost_coeff = 0.0
finish_ahead_of_time_reward_coeff = 0.1

if __name__ == "__main__":
    block_positions, belief = get_positions_and_init_belief()

    agent = Agent(initial_blocks_position_belief=belief,
                  max_steps=max_steps,
                  stacking_reward=stacking_reward,
                  cost_coeff=cost_coeff,
                  finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff)

    actual_init_state = State(steps_left=max_steps,
                              block_positions=block_positions)
    env = Environment.from_agent(agent=agent, init_state=actual_init_state)

    planner = pomdp_py.POUCT(max_depth=5,
                             planning_time=3,
                             rollout_policy=agent.policy_model,
                             show_progress=True)

    total_reward = 0
    plot_all_blocks_beliefs(agent.belief)
    for i in range(max_steps):
        actual_action = planner.plan(agent)
        actual_reward = env.state_transition(actual_action, execute=True)
        actual_observation = env.provide_observation(agent.observation_model, actual_action)

        total_reward += actual_reward

        planner.update(agent, actual_action, actual_observation)
        # planner.update only updates the tree, but not the belief and history of the agent!
        # I implemented agent.update to do both:
        agent.update(actual_action, actual_observation)

        print("------------")
        print(f"step: {i}, action: {actual_action}, observation: {actual_observation}, reward: {actual_reward}")
        print(f"total reward: {total_reward}, hidden_state={env.state}")
        print("num_sims", planner.last_num_sims)
        plot_all_blocks_beliefs(agent.belief)

        # TODO: plot actual block positions,
        #   pickup attempt points,
        #   negative and positive sensing points
        # TODO: Run multiple times to find more bugs
        # TODO: sample that filters and doesn't try to find other valid points
        # TODO: think how not to call get_all_actions on each rollout
        # TODO: resampling happens alot. print it in gaussian masked distribution again
        # TODO: profile
