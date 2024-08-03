import pomdp_py
from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform, workspace_x_lims_default, \
    workspace_y_lims_default, sample_block_positions_from_dists
from lab_ur_stack.utils.workspace_utils import goal_tower_position
from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief
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
    sigmas = [[0.05, 0.07], [0.1, 0.08], [0.07, 0.07], [0.15, 0.15], [0.02, 0.03]]
    sigmas = sigmas[:3]
    belief = BeliefModel(3, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

    block_positions = sample_block_positions_from_dists(belief.block_beliefs)

    return block_positions, belief


max_steps = 20

stacking_reward = 1
sensing_cost_coeff = 0.0
stacking_cost_coeff = 0.0
finish_ahead_of_time_reward_coeff = 0.1

n_blocks_for_actions = 2
points_to_sample_for_each_block = 150
sensing_actions_to_sample_per_block = 2
max_planning_depth = 6
tower_pos = goal_tower_position

if __name__ == "__main__":
    block_positions, belief = get_positions_and_init_belief()

    agent = Agent(initial_blocks_position_belief=belief,
                  max_steps=max_steps,
                  tower_position=tower_pos,
                  stacking_reward=stacking_reward,
                  sensing_cost_coeff=sensing_cost_coeff,
                  stacking_cost_coeff=stacking_cost_coeff,
                  finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff,
                  n_blocks_for_actions=n_blocks_for_actions,
                  points_to_sample_for_each_block=points_to_sample_for_each_block,
                  sensing_actions_to_sample_per_block=sensing_actions_to_sample_per_block)

    actual_init_state = State(steps_left=max_steps,
                              block_positions=block_positions,
                              robot_position=tower_pos,)
    env = Environment.from_agent(agent=agent, init_state=actual_init_state)

    planner = pomdp_py.POUCT(max_depth=max_planning_depth,
                             # planning_time=300,
                             num_sims=2000,
                             discount_factor=1.0,
                             rollout_policy=agent.policy_model,
                             show_progress=True)

    print("init state", actual_init_state)
    plot_all_blocks_beliefs(agent.belief,
                            actual_states=block_positions, )

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
                positive_sensing_points.append((actual_action.x, actual_action.y))
            else:
                negative_sensing_points.append((actual_action.x, actual_action.y))
        elif isinstance(actual_observation, ObservationStackAttemptResult):
            pickup_attempt_points.append((actual_action.x, actual_action.y))
        plot_all_blocks_beliefs(agent.belief,
                                actual_states=block_positions,
                                negative_sensing_points=negative_sensing_points,
                                positive_sensing_points=positive_sensing_points,
                                pickup_attempt_points=pickup_attempt_points)

        pass

        # TODO: next steps:
        # TODO: actions for not all blocks... (sensed positie, low variance)
        # TODO: Less actions (up to two blocks, will go deeper + faster get all actions)
        # TODO: Test effects of less samples. maybe it has no effect on lab computer

        # TODO: don't sample pickup if no masks or bounds?
        # TODO: action prior for sensing
        # TODO: Use TreeDebuger or visualization
        # TODO: learning? belief will belong to helper only
