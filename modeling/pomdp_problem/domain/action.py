import pomdp_py


class ActionBase(pomdp_py.Action):
    def __init__(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y

    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        return hash((self.x, self.y))


class ActionAttemptStack(ActionBase):
    def __init__(self, x, y):
        super(ActionBase).__init__(x, y)

    def __eq__(self, other):
        return isinstance(other, ActionAttemptStack) and self.x == other.x and self.y == other.y


class ActionSense(ActionBase):
    def __init__(self, x, y):
        super(ActionBase).__init__(x, y)

    def __eq__(self, other):
        return isinstance(other, ActionSense) and self.x == other.x and self.y == other.y
