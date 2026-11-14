from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase

from algo.base import SharedMemory


class Frontier(SharedMemory):
    def reset(self) -> None:
        env = self.env
        _, height, width = env.walls.shape
        shape = (env.num_envs, self.n_owners, height, width)
        self.known = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.visited = torch.zeros_like(self.known)
        self.dist = torch.full(shape, torch.inf, device=env.device)
        self.dirty = torch.ones(shape[:2], dtype=torch.bool, device=env.device)

    def observe(self) -> None:
        env = self.env
        before = self.known.clone(), self.visited.clone()
        offsets = env.directions[:, None] * env.distances[:, None]
        cells = env.positions[:, :, None, None] + offsets
        rows, cols = cells[..., 0], cells[..., 1]
        e = self.env_index[..., None, None].expand_as(rows)
        o = self.owner[None, :, None, None].expand_as(rows)
        walls = env.walls[e, rows, cols].to(torch.uint8)
        hidden = torch.cat([torch.zeros_like(walls[..., :1]), walls[..., :-1]], -1).cummax(-1).values.bool()
        visible = ~hidden
        self.known[e[visible], o[visible], rows[visible], cols[visible]] = True

        rows, cols = env.positions.unbind(-1)
        self.known[self.env_index, self.owner, rows, cols] = True
        self.visited[self.env_index, self.owner, rows, cols] = True
        changed = (self.known != before[0]) | (self.visited != before[1])
        self.dirty |= changed.flatten(2).any(-1)

        lost = torch.isinf(self.dist[self.env_index, self.owner, rows, cols])
        self.dirty[self.env_index.expand_as(lost)[lost], self.owner.expand_as(lost)[lost]] = True

    def distances(self) -> torch.Tensor:
        if not self.dirty.any():
            return self.dist
        idx = self.dirty.nonzero(as_tuple=True)
        passable = self.known[idx] & ~self.env.walls[idx[0]]
        dist = torch.where(passable & ~self.visited[idx], 0.0, torch.inf)

        row = torch.full(self.dirty.shape, -1, device=self.env.device)
        row[idx] = torch.arange(len(idx[0]), device=self.env.device)
        agent_rows = row[self.env_index, self.owner]
        tracked = agent_rows >= 0
        agent_rows = agent_rows[tracked]
        rows, cols = self.env.positions[tracked].unbind(-1)

        for _ in range(passable[0].numel()):
            p = F.pad(dist, (1, 1, 1, 1), value=torch.inf)
            shifted = torch.minimum(
                torch.minimum(p[:, :-2, 1:-1], p[:, 2:, 1:-1]),
                torch.minimum(p[:, 1:-1, :-2], p[:, 1:-1, 2:]),
            ) + 1
            new = torch.where(passable, torch.minimum(dist, shifted), torch.inf)
            changed = (new != dist).flatten(1).any(-1)
            dist = new
            if (~torch.isinf(dist[agent_rows, rows, cols]) | ~changed[agent_rows]).all():
                break
        self.dist[idx] = dist
        self.dirty[:] = False
        return self.dist

    def decide(self, td: TensorDictBase) -> torch.Tensor:
        self.observe()
        dist = self.distances()
        cells = self.neighbours()
        e, o = self.env_index[..., None], self.owner[None, :, None]
        cost = dist[e, o, cells[..., 0], cells[..., 1]] + 0.5 * self.rand(*cells.shape[:-1])
        rows, cols = self.env.positions.unbind(-1)
        lost = torch.isinf(dist[self.env_index, self.owner, rows, cols])
        return torch.where(lost, 0, self.to_action(cost.argmin(-1)))
