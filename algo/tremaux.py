from __future__ import annotations

import torch
from tensordict import TensorDictBase

from algo.base import SharedMemory


class Tremaux(SharedMemory):
    def reset(self) -> None:
        env = self.env
        _, height, width = env.walls.shape
        shape = (env.num_envs, self.n_owners, height, width)
        self.marks = torch.zeros(*shape, 4, dtype=torch.int16, device=env.device)
        self.visits = torch.zeros(shape, dtype=torch.int32, device=env.device)
        self.previous = env.positions.clone()
        self.turn_back = torch.zeros(env.num_envs, env.n_agents, dtype=torch.bool, device=env.device)
        rows, cols = env.positions.unbind(-1)
        self.visits[self.env_index, self.owner, rows, cols] = 1

    def remember_moves(self) -> None:
        env, e, o = self.env, self.env_index.expand(-1, self.env.n_agents), self.owner.expand_as(self.env.headings)
        moved = (env.positions != self.previous).any(-1)
        rows, cols = env.positions.unbind(-1)
        entry = (env.headings + 2) % 4
        arrived_before = self.visits[e, o, rows, cols] > 0

        index = (e[moved], o[moved])
        one = torch.ones(len(index[0]), dtype=self.marks.dtype, device=env.device)
        self.marks.index_put_((*index, *self.previous[moved].unbind(-1), env.headings[moved]), one, accumulate=True)
        self.marks.index_put_((*index, rows[moved], cols[moved], entry[moved]), one, accumulate=True)
        self.visits.index_put_((*index, rows[moved], cols[moved]), one.int(), accumulate=True)

        first_pass = self.marks[e, o, rows, cols, entry] == 1
        self.turn_back = torch.where(moved, arrived_before & first_pass, self.turn_back)
        self.previous = env.positions.clone()

    def decide(self, td: TensorDictBase) -> torch.Tensor:
        self.remember_moves()
        env = self.env
        rows, cols = env.positions.unbind(-1)
        marks = self.marks[self.env_index, self.owner, rows, cols]
        entry = (env.headings + 2) % 4
        is_entry = torch.arange(4, device=env.device) == entry[..., None]

        cost = marks + 0.5 * ~is_entry + 0.1 * self.rand(*marks.shape)
        cost = torch.where(self.free_directions() & (marks < 2), cost, torch.inf)
        direction = torch.where(self.turn_back, entry, cost.argmin(-1))
        stuck = torch.isinf(cost).all(-1) & ~self.turn_back
        return torch.where(stuck, 0, self.to_action(direction))
