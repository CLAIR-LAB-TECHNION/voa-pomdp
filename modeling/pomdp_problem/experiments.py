import pomdp_py
from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform, workspace_x_lims_default, \
    workspace_y_lims_default, sample_block_positions_from_dists
from modeling.belief.block_position_belief import BlocksPositionsBelief
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.models.policy_model import BeliefModel
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack
from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from pomdp_py.utils import TreeDebugger


def get_positions_and_init_belief():
    mus = [[-0.85, -0.9], [-0.75, -0.75], [-0.65, -0.65]]
    sigmas = [[0.05, 0.2], [0.25, 0.08], [0.1, 0.15], [0.15, 0.15], [0.02, 0.03]]
    sigmas = sigmas[:3]
    belief = BeliefModel(3, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

    block_positions = sample_block_positions_from_dists(belief.block_beliefs)

    return block_positions, belief


max_steps = 10
stacking_reward = 1
cost_coeff = 0.0
finish_ahead_of_time_reward_coeff = 0.1
points_to_sample_for_each_block = 200

if __name__ == "__main__":
    block_positions, belief = get_positions_and_init_belief()

    agent = Agent(initial_blocks_position_belief=belief,
                  max_steps=max_steps,
                  stacking_reward=stacking_reward,
                  cost_coeff=cost_coeff,
                  finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff,
                  points_to_sample_for_each_block=points_to_sample_for_each_block)

    actual_init_state = State(steps_left=max_steps,
                              block_positions=block_positions)
    env = Environment.from_agent(agent=agent, init_state=actual_init_state)

    planner = pomdp_py.POUCT(max_depth=8,
                             # planning_time=300,
                             num_sims=500,
                             discount_factor=1.0,
                             rollout_policy=agent.policy_model,
                             show_progress=True)

    print("init state", actual_init_state)
    plot_all_blocks_beliefs(agent.belief,
                            actual_states=block_positions,)

    total_reward = 0
    positive_sensing_points = []
    negative_sensing_points = []
    pickup_attempt_points = []
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
        dd = TreeDebugger(agent.tree)
        print("num_sims", planner.last_num_sims, "max depth reached", dd.depth)
        if isinstance(actual_observation, ObservationSenseResult):
            if actual_observation.is_occupied:
                positive_sensing_points.append((actual_observation.x, actual_observation.y))
            else:
                negative_sensing_points.append((actual_observation.x, actual_observation.y))
        elif isinstance(actual_observation, ObservationStackAttemptResult):
            pickup_attempt_points.append((actual_action.x, actual_action.y))
        plot_all_blocks_beliefs(agent.belief,
                                actual_states=block_positions,
                                negative_sensing_points=negative_sensing_points,
                                positive_sensing_points=positive_sensing_points,
                                pickup_attempt_points=pickup_attempt_points)

        pass
        # TODO next step is to implement new beleif representation that doesn't resolve overlap and doesn't compute
        #   normalization factor. Add flag to current belief
        # TODO we are plotting many cases where we stuck in resampling where block support is really small.
        #   In that case sample on a grid within this area, need to maintain some kind of new bounds

        # TODO: Use TreeDebuger or visualization
        # TODO: sample that filters and doesn't try to find other valid points
        # TODO: actions for not all blocks... (sensed positie, low variance)
        # TODO: Save amount of times blocks sensed positive and priortize actions for that block
        # TODO: Run multiple times to find more bugs
        # TODO: Run on lab computer and check it/s
        # TODO: think how not to call get_all_acti ons on each rollout
        # TODO: Is there a way to limit max_num_rollouts per node?
        # TODO: profile
        # TODO: lab task for tomorrow check pickup success in env
        # TODO: learning? belief will belong to helper only
