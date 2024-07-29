import pomdp_py


class ActionBase(pomdp_py.Action):
    def __init__(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y

    def __eq__(self, other):
        raise NotImplementedError

class ActionAttemptStack(ActionBase):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __eq__(self, other):
        return isinstance(other, ActionAttemptStack) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(("attempt_stack", self.x, self.y))

    def __str__(self):
        return f"AttemptStack({self.x}, {self.y})"


class ActionSense(ActionBase):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __eq__(self, other):
        return isinstance(other, ActionSense) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(("sense", self.x, self.y))

    def __str__(self):
        return f"Sense({self.x}, {self.y})"
