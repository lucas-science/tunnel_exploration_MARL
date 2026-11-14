from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDictBase

from algo.base import Algorithm
from tunnel_env import ACTIONS, GROUP, TunnelEnv


class WallFollower(Algorithm):
    def __init__(self, env: TunnelEnv, seed: int = 0, side: Literal["left", "right"] = "left") -> None:
        super().__init__(env, seed)
        other = "right" if side == "left" else "left"
        order = [ACTIONS.index(a) for a in (side, "forward", other, "back")]
        self.order = torch.tensor(order, device=env.device)
        self.side = ACTIONS.index(side)

    def reset(self) -> None:
        self.on_wall = torch.zeros_like(self.env.headings, dtype=torch.bool)

    def decide(self, td: TensorDictBase) -> torch.Tensor:
        mask = td[GROUP, "action_mask"]
        forward = ACTIONS.index("forward")
        self.on_wall |= ~mask[..., self.side] | ~mask[..., forward]
        free = mask[..., self.order]
        actions = torch.where(free.any(-1), self.order[free.float().argmax(-1)], 0)
        return torch.where(self.on_wall, actions, forward)
