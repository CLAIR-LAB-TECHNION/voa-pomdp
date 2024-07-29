import pomdp_py


class ObservationBase(pomdp_py.Observation):
    def __init__(self, steps_left):
        self.steps_left = steps_left  # this is always fully observable

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


class ObservationSenseResult(ObservationBase):
    def __init__(self, x, y, is_occupied, steps_left):
        super().__init__(steps_left)
        self.x = x
        self.y = y
        self.is_occupied = is_occupied

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.is_occupied == other.is_occupied\
            and self.steps_left == other.steps_left

    def __hash__(self):
        return hash((self.x, self.y, self.is_occupied, self.steps_left))

    def __str__(self):
        return f"(sense_res, x={self.x}, y={self.y}, is_occupied={self.is_occupied}, steps_left={self.steps_left})"


class ObservationStackAttemptResult(ObservationBase):
    def __init__(self, is_object_picked, steps_left):
        super().__init__(steps_left)
        self.is_object_picked = is_object_picked

    def __eq__(self, other):
        return self.is_object_picked == other.is_object_picked and self.steps_left == other.steps_left

    def __hash__(self):
        return hash((self.is_object_picked, self.steps_left))

    def __str__(self):
        return f"(is_object_picked={self.is_object_picked}, steps_left={self.steps_left})"