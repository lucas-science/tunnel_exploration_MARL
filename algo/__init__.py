from __future__ import annotations

from functools import partial

from algo.base import Algorithm
from algo.frontier import Frontier
from algo.random_walk import ForwardRandomWalk, RandomWalk
from algo.tremaux import Tremaux
from algo.wall_follower import WallFollower

ALGORITHMS = {
    "random": RandomWalk,
    "random_forward": ForwardRandomWalk,
    "left_hand": partial(WallFollower, side="left"),
    "right_hand": partial(WallFollower, side="right"),
    "tremaux": Tremaux,
    "tremaux_shared": partial(Tremaux, shared=True),
    "frontier": Frontier,
    "frontier_shared": partial(Frontier, shared=True),
}

__all__ = ["ALGORITHMS", "Algorithm"]
