import pomdp_py
import numpy as np
from modeling.pomdp.pomdp_py_domain import State, Observation, Action


class ObservationModel(pomdp_py.ObservationModel):
    def __init__