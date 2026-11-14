from __future__ import annotations

import torch
from tensordict import TensorDictBase

from algo.base import Algorithm
from tunnel_env import GROUP


class RandomWalk(Algorithm):
    def decide(self, td: TensorDictBase) -> torch.Tensor:
        return self.random_move(td)


class ForwardRandomWalk(Algorithm):
    def decide(self, td: TensorDictBase) -> torch.Tensor:
        moves = td[GROUP, "action_mask"][..., 1:]
        priority = torch.tensor([2.0, 1.0, 1.0, 0.0], device=moves.device)
        scores = torch.where(moves, priority + self.rand(*moves.shape), -1.0)
        return torch.where(moves.any(-1), scores.argmax(-1) + 1, 0)
