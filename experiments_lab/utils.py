from modeling.belief.belief_plotting import plot_all_blocks_beliefs
from modeling.pomdp_problem.domain.observation import ObservationSenseResult, ObservationStackAttemptResult


def plot_belief_with_history(current_belief,
                             actual_state=None,
                             history=[],
                             observed_mus_and_sigmas=None,
                             ret_as_image=True):
    positive_sens = []
    negative_sens = []
    failed_pickups = []
    for a, o in history:
        if isinstance(o, ObservationSenseResult):
            if o.is_occupied:
                positive_sens.append((a.x, a.y))
            else:
                negative_sens.append((a.x, a.y))
        elif isinstance(o, ObservationStackAttemptResult):
            if not o.is_object_picked:
                failed_pickups.append((a.x, a.y))

    return plot_all_blocks_beliefs(current_belief,
                                   actual_states=actual_state,
                                   positive_sensing_points=positive_sens,
                                   negative_sensing_points=negative_sens,
                                   pickup_attempt_points=failed_pickups,
                                   per_block_observed_mus_and_sigmas=observed_mus_and_sigmas,
                                   ret_as_image=ret_as_image)