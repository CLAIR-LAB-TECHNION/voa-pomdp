""" Utils to work with pomdp_py particle belief """
from pomdp_py import Particles
from modeling.pomdp_problem.domain.state import State
from modeling.belief.block_position_belief import BlocksPositionsBelief
from lab_ur_stack.utils.workspace_utils import goal_tower_position, sample_block_positions_from_dists_vectorized


def explicit_belief_to_particles(explicit_belief: BlocksPositionsBelief,
                                 num_particles,
                                 observable_steps_left,
                                 observable_robot_position=goal_tower_position,
                                 observable_last_stack_attempt_succeded=None) -> Particles:
    """
     Convert an explicit belief to a particle belief will return particles where each state is already POMDP
     problem state. That will require to specify the observable parts of the states to build them
     """
    all_block_positions = sample_block_positions_from_dists_vectorized(
        explicit_belief.block_beliefs,
        num_particles
    )

    states = [
        State(
            block_positions=block_positions,
            steps_left=observable_steps_left,
            robot_position=observable_robot_position,
            last_stack_attempt_succeded=observable_last_stack_attempt_succeded
        )
        for block_positions in all_block_positions
    ]

    return Particles(states)