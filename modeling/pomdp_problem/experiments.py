import pomdp_py
from lab_ur_stack.utils.workspace_utils import sample_block_positions_uniform, workspace_x_lims_default, \
    workspace_y_lims_default, sample_block_positions_from_dists
from lab_ur_stack.utils.workspace_utils import goal_tower_position
from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief
from modeling.policies.pouct_planner_policy import POUCTPolicy
from modeling.pomdp_problem.agent.agent import Agent
from modeling.pomdp_problem.env.env import Environment
from modeling.pomdp_problem.models.policy_model import BeliefModel
from modeling.pomdp_problem.domain.state import State
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult, \
    ObservationReachedTerminal
from modeling.pomdp_problem.domain.action import ActionSense, ActionAttemptStack
from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from pomdp_py.utils import TreeDebugger
from modeling.policies.hand_made_policy import HandMadePolicy


# def get_positions_and_init_belief():
#     mus = [[-0.85, -0.9], [-0.75, -0.75], [-0.65, -0.65], [-0.6, -0.9]]
#     sigmas = [[0.05, 0.07], [0.1, 0.08], [0.07, 0.07], [0.15, 0.15], [0.02, 0.03]]
#     sigmas = sigmas[:4]
#     belief = BeliefModel(4, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)
#
#     block_positions = sample_block_positions_from_dists(belief.block_beliefs)
#
#     return block_positions, belief

def get_positions_and_init_belief():
    mus = [[-0.5555811001989546, -0.8685714370415379], [-0.5510296397412436, -0.5921838671142652],
           [-0.8559652664168667, -0.8528636466344859], [-0.784730232877579, -0.5888542174800786]]
    sigmas = [[0.14044278905235758, 0.11851284931192309], [0.03893375750787616, 0.0802298602164658],
              [0.13037169026799353, 0.03926828321910571], [0.11914382647817032, 0.08134558114274028]]

    belief = BeliefModel(4, workspace_x_lims_default, workspace_y_lims_default, mus, sigmas)

    block_positions = [[-0.6225174767780174, -0.9063979509309235], [-0.5818126962048723, -0.684841526937952],
                       [-0.6989217892322716, -0.8132851505436323], [-0.8356209326158366, -0.5579876134076684]]

    return block_positions, belief


max_steps = 20

stacking_reward = 1
sensing_cost_coeff = 0.05
stacking_cost_coeff = 0.05  # stack takes much more time than sensing
finish_ahead_of_time_reward_coeff = 0.1

num_sims=1000
n_blocks_for_actions = 2
points_to_sample_for_each_block = 50
sensing_actions_to_sample_per_block = 2
max_planning_depth = 5
tower_pos = goal_tower_position

if __name__ == "__main__":
    block_positions, belief = get_positions_and_init_belief()

    policy = POUCTPolicy(initial_belief=belief,
                         max_steps=max_steps,
                         tower_position=tower_pos,
                         max_planning_depth=max_planning_depth,
                         stacking_reward=stacking_reward,
                         sensing_cost_coeff=sensing_cost_coeff,
                         stacking_cost_coeff=stacking_cost_coeff,
                         finish_ahead_of_time_reward_coeff=finish_ahead_of_time_reward_coeff,
                         n_blocks_for_actions=n_blocks_for_actions,
                         points_to_sample_for_each_block=points_to_sample_for_each_block,
                         sensing_actions_to_sample_per_block=sensing_actions_to_sample_per_block,
                         num_sims=1000,
                         show_progress=True
                         )
    policy.reset(belief)

    agent = policy.agent

    actual_init_state = State(steps_left=max_steps,
                              block_positions=block_positions,
                              robot_position=tower_pos, )
    env = Environment.from_agent(agent=agent, init_state=actual_init_state)

    print("init state", actual_init_state)
    plot_all_blocks_beliefs(agent.belief,
                            actual_states=block_positions, )

    total_reward = 0
    positive_sensing_points = []
    negative_sensing_points = []
    pickup_attempt_points = []
    planning_times = []
    for i in range(max_steps):
        # actual_action = planner.plan(agent)
        actual_action = policy.sample_action(agent.belief, agent.history)
        actual_reward = env.state_transition(actual_action, execute=True)
        actual_observation = env.provide_observation(agent.observation_model, actual_action)

        total_reward += actual_reward

        # dd = TreeDebugger(agent.tree)
        # dd.p(6)

        if isinstance(actual_observation, ObservationReachedTerminal):
            break

        # planner.update(agent, actual_action, actual_observation)
        # planner.update only updates the tree, but not the belief and history of the agent!
        # I implemented agent.update to do both:
        agent.update(actual_action, actual_observation)

        print("------------")
        print(f"step: {i}, action: {actual_action}, observation: {actual_observation}, reward: {actual_reward}")
        print(f"total reward: {total_reward}, hidden_state={env.state}")
        # print("num_sims", planner.last_num_sims, "max depth reached", dd.depth)
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

        planning_times.append(policy.planner.last_planning_time)

        pass
