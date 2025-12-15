from collections import namedtuple
import numpy as np

Transition = namedtuple(
    "Transition",
    [
        "state",
        "obs",
        "action",
        "reward",
        "next_state",
        "next_obs",
        "done",
    ],
)

TransitionBatch = namedtuple(
    "TransitionBatch",
    [
        "state",
        "obs",
        "action",
        "reward",
        "next_state",
        "next_obs",
        "done",
    ],
)
