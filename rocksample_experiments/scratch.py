import pomdp_py

from rocksample_experiments.RSRolloutPolicy import CustomRSPolicyModel
from rocksample_experiments.rocksample_problem import RockSampleProblem, init_particles_belief



def run_rocksample_experiment():
    # Problem parameters
    n = 7  # Grid size (nxn)
    k = 8  # Number of rocks
    num_steps = 100
    n_sims = 5000
    max_depth = 20
    discount = 0.95
    exploration_const = 5
    half_efficiency_dist = 20

    # Add statistics tracking
    stats = {
        'good_rocks': 0,
        'bad_rocks': 0,
        'good_samples': 0,
        'bad_samples': 0,
        'sense_actions': 0
    }

    # Generate initial state and rock locations
    init_state, rock_locs = RockSampleProblem.generate_instance(n, k)

    # Count initial rocks
    for rock in init_state.rocktypes:
        if rock == 'good':
            stats['good_rocks'] += 1
        else:
            stats['bad_rocks'] += 1

    # Initialize belief with particles
    init_belief = init_particles_belief(
        k=k,
        num_particles=200,
        init_state=init_state,
        belief="uniform"
    )

    # Create RockSample problem instance
    rocksample = RockSampleProblem(
        n=n,
        k=k,
        init_state=init_state,
        rock_locs=rock_locs,
        init_belief=init_belief,
        half_efficiency_dist=half_efficiency_dist
    )

    preffered_actions_rollout_policy = CustomRSPolicyModel(n, k, rock_locs)

    # Initialize POMCP planner
    pomcp = pomdp_py.POMCP(
        num_sims=n_sims,
        max_depth=max_depth,
        discount_factor=discount,
        exploration_const=exploration_const,
        rollout_policy=preffered_actions_rollout_policy,
        num_visits_init=1,
        show_progress=True
    )

    print("Initial state:")
    rocksample.print_state()

    # Run simulation
    total_reward = 0
    total_discounted_reward = 0
    gamma = 1.0

    for step in range(num_steps):
        print(f"\nStep {step + 1}")

        # Plan action
        action = pomcp.plan(rocksample.agent)
        print("planning time:", pomcp.last_planning_time)

        # Execute action and get reward
        reward = rocksample.env.state_transition(action, execute=True)

        # Track statistics based on action and reward
        if isinstance(action, str):
            action_name = action
        else:
            action_name = action.name if hasattr(action, 'name') else str(action)

        if "sample" in action_name.lower():
            if reward > 0:  # Reward is positive for sampling good rocks
                stats['good_samples'] += 1
            elif reward < 0:  # Reward is negative for sampling bad rocks
                stats['bad_samples'] += 1
        elif "check" in action_name.lower():  # Check actions
            stats['sense_actions'] += 1

        # Get observation
        observation = rocksample.env.provide_observation(
            rocksample.agent.observation_model,
            action
        )

        # Update agent history and planner
        rocksample.agent.update_history(action, observation)
        pomcp.update(rocksample.agent, action, observation)

        # Update rewards
        total_reward += reward
        total_discounted_reward += reward * gamma
        gamma *= discount

        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        print(f"Total Discounted Reward: {total_discounted_reward}")
        rocksample.print_state()

        if rocksample.in_exit_area(rocksample.env.state.position):
            print("\nReached exit area! Terminating...")
            break

    print("\nFinal Statistics:")
    print(f"Initial rock distribution:")
    print(f"  Good rocks: {stats['good_rocks']}")
    print(f"  Bad rocks: {stats['bad_rocks']}")
    print(f"Actions taken:")
    print(f"  Good rocks sampled: {stats['good_samples']}")
    print(f"  Bad rocks sampled: {stats['bad_samples']}")
    print(f"  Sense actions used: {stats['sense_actions']}")

    return total_reward, total_discounted_reward


if __name__ == "__main__":
    total_reward, total_discounted_reward = run_rocksample_experiment()
    print(f"\nFinal Results:")
    print(f"Total Reward: {total_reward}")
    print(f"Total Discounted Reward: {total_discounted_reward}")