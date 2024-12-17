"""
 Rollout policy for rock sample problem that applies knowledge of the problem to make better decisions.
 This is similar to the original one that was used in the POMCP paper.
"""
import random

import numpy as np
from rocksample_experiments.rocksample_problem import RSPolicyModel, CheckAction, MoveEast, MoveNorth, \
    MoveSouth, MoveWest, SampleAction, State, RockType
from line_profiler_pycharm import profile


class CustomRSPolicyModel(RSPolicyModel):
    def __init__(self, n, k, rock_locs):
        super().__init__(n, k)
        self.rock_locs = rock_locs
        # rock locs is a map from position to rock id, create reverse map as an array of positions
        self.rock_pos = np.zeros((k, 2))
        for key, value in rock_locs.items():
            self.rock_pos[value] = key

    @profile
    def rollout(self, state: State, history=None):
        # Handle no history case with parent's random policy
        if history is None:
            return super().rollout(state, history)

        actions = []

        # Check if there's a rock at current position worth sampling
        is_at_rock = (state.position[0] == self.rock_pos[:, 0])  & (state.position[1] == self.rock_pos[:, 1])
        current_rock = None if not is_at_rock.any() else np.where(is_at_rock)[0][0]

        # if there is an uncollected rock, and it was observed to be good more than bad, sample it
        if current_rock is not None:
            collected = False
            total = 0
            for action, observation in reversed(history):  # iterate backwards to find earlier if collected
                if isinstance(action, SampleAction) and action.position[0] == state.position[0] and \
                        action.position[1] == state.position[1]:
                    collected = True
                    break
                if isinstance(action, CheckAction) and action.rock_id == current_rock:
                    if observation.quality == RockType.GOOD:
                        total += 1
                    elif observation.quality == RockType.BAD:
                        total -= 1
            if not collected and total > 0:
                return SampleAction(state.position)

        # Process rocks to determine interesting directions
        per_rock_totals = np.zeros(len(self.rock_locs))
        for action, observation in history:
            if isinstance(action, SampleAction):
                # Sampled rocks are not interesting
                if action.position in self.rock_locs.keys():
                    per_rock_totals[self.rock_locs[action.position]] = -100
            elif isinstance(action, CheckAction):
                if observation.quality == RockType.GOOD:
                    per_rock_totals[action.rock_id] += 1
                elif observation.quality == RockType.BAD:
                    per_rock_totals[action.rock_id] -= 1

        interesting_rock_ids = np.where(per_rock_totals >= 0)[0]
        all_bad = len(interesting_rock_ids) == 0

        # If all rocks seem bad, go east
        if all_bad:
            return MoveEast

        interesting_positions = self.rock_pos[interesting_rock_ids]
        north_interesting = (interesting_positions[:, 1] > state.position[1]).any()
        south_interesting = (interesting_positions[:, 1] < state.position[1]).any()
        east_interesting = (interesting_positions[:, 0] > state.position[0]).any()
        west_interesting = (interesting_positions[:, 0] < state.position[0]).any()

        # Add movement actions only in interesting directions
        if north_interesting and state.position[1] < self._n - 1:
            actions.append(MoveNorth)
        if south_interesting and state.position[1] > 0:
            actions.append(MoveSouth)
        if east_interesting and state.position[0] < self._n - 1:
            actions.append(MoveEast)
        if west_interesting and state.position[0] > 0:
            actions.append(MoveWest)

        # add check actions only interesting rocks that haven't been checked in the last 3 steps
        # (interesting = not sampled, checked more good than bad or not checked at all)
        # This is slightly different from the original POMCP paper implementation which have access to rock probability
        # of being good in the state.
        for rock_id in interesting_rock_ids:
            if not any(isinstance(action, CheckAction) and action.rock_id == rock_id for action, _ in history[-3:]):
                actions.append(CheckAction(rock_id))

        if not actions:
            return MoveEast if state.position[0] < self._n - 1 else random.sample(self.get_all_actions(state=state), 1)[
                0]

        return random.choice(actions)
