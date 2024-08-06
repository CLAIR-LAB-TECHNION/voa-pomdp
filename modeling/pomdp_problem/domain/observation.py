import pomdp_py


class ObservationBase(pomdp_py.Observation):
    def __init__(self, robot_position, steps_left):
        self.steps_left = steps_left  # this is always fully observable
        self.robot_position = robot_position

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


class ObservationSenseResult(ObservationBase):
    def __init__(self, is_occupied, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_occupied = is_occupied

    def __eq__(self, other):
        return self.is_occupied == other.is_occupied and self.robot_position[0] == other.robot_position[0]\
             and self.robot_position[1] == other.robot_position[1] and self.steps_left == other.steps_left

    def __hash__(self):
        return hash((self.robot_position[0], self.robot_position[1], self.is_occupied, self.steps_left))

    def __str__(self):
        return f"(sense_res, robot_position={self.robot_position}," \
               f" is_occupied={self.is_occupied}, steps_left={self.steps_left})"


class ObservationStackAttemptResult(ObservationBase):
    def __init__(self, is_object_picked, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_object_picked = is_object_picked

    def __eq__(self, other):
        return self.is_object_picked == other.is_object_picked and self.steps_left == other.steps_left\
            and self.robot_position[0] == other.robot_position[0] and self.robot_position[1] == other.robot_position[1]

    def __hash__(self):
        return hash((self.robot_position[0], self.robot_position[1], self.is_object_picked, self.steps_left))

    def __str__(self):
        return f"(attempt_stack_res, robot_position={self.robot_position}, " \
               f"is_object_picked={self.is_object_picked}, steps_left={self.steps_left})"


class EpisodeEndObservation(ObservationBase):
    def __init__(self):
        super().__init__(robot_position=None, steps_left=0)

    def __eq__(self, other):
        return isinstance(other, EpisodeEndObservation)

    def __hash__(self):
        return hash("EpisodeEndObservation")

    def __str__(self):
        return "EpisodeEndObservation"
