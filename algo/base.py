from __future__ import annotations

import torch
from tensordict import TensorDictBase

from tunnel_env import GROUP, TunnelEnv

OFFSET_TO_ACTION = torch.tensor([1, 3, 4, 2])
ESCAPE_PROBABILITY = 0.5


class Algorithm:
    def __init__(self, env: TunnelEnv, seed: int = 0) -> None:
        self.env = env
        self.rng = torch.Generator().manual_seed(seed)
        self.env_index = env.env_index[:, None]
        self.agents = torch.arange(env.n_agents, device=env.device)

    def reset(self) -> None:
        pass

    def decide(self, td: TensorDictBase) -> torch.Tensor:
        raise NotImplementedError

    def start(self) -> None:
        self.reset()
        self.last_actions = torch.zeros_like(self.env.headings)
        self.last_positions = self.env.positions.clone()

    def act(self, td: TensorDictBase) -> torch.Tensor:
        actions = self.decide(td)
        blocked = (self.last_actions > 0) & (self.env.positions == self.last_positions).all(-1)
        escape = blocked & (self.rand(*blocked.shape) < ESCAPE_PROBABILITY)
        actions = torch.where(escape, self.random_move(td), actions)
        self.last_actions, self.last_positions = actions, self.env.positions.clone()
        return actions

    def random_move(self, td: TensorDictBase) -> torch.Tensor:
        moves = td[GROUP, "action_mask"][..., 1:]
        scores = torch.where(moves, self.rand(*moves.shape), -1.0)
        return torch.where(moves.any(-1), scores.argmax(-1) + 1, 0)

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.rng).to(self.env.device)

    def neighbours(self) -> torch.Tensor:
        return self.env.positions[:, :, None] + self.env.directions

    def free_directions(self) -> torch.Tensor:
        cells = self.neighbours()
        return ~self.env.walls[self.env_index[..., None], cells[..., 0], cells[..., 1]]

    def to_action(self, direction: torch.Tensor) -> torch.Tensor:
        offset = (direction - self.env.headings) % 4
        return OFFSET_TO_ACTION.to(direction.device)[offset]


class SharedMemory(Algorithm):
    def __init__(self, env: TunnelEnv, seed: int = 0, shared: bool = False) -> None:
        super().__init__(env, seed)
        self.owner = torch.zeros_like(self.agents) if shared else self.agents
        self.n_owners = 1 if shared else env.n_agents
