from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import ParallelEnv

from multi_agent_RL.tunnel_map import TunnelMap

AgentID = str
ObsType = npt.NDArray[np.float32]

MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
DIRECTIONS = MOVES[1:]
N_CELL_FEATURES = 3


class TunnelExplorationEnv(ParallelEnv[AgentID, ObsType, int]):
    metadata = {"name": "tunnel_exploration_v0", "render_modes": ["ansi"]}

    def __init__(
        self,
        maps: TunnelMap | Sequence[TunnelMap],
        n_agents: int = 4,
        view_range: int = 3,
        max_steps: int = 500,
        step_penalty: float = 0.01,
        reward_mode: Literal["grouped", "mixed"] = "grouped",
        reward_alpha: float = 0.5,
        render_mode: str | None = None,
    ) -> None:
        self.maps = [maps] if isinstance(maps, TunnelMap) else list(maps)
        self.view_range = view_range
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.reward_mode = reward_mode
        self.reward_alpha = reward_alpha
        self.render_mode = render_mode

        self.possible_agents = [f"drone_{i}" for i in range(n_agents)]
        self._agent_index = {a: i for i, a in enumerate(self.possible_agents)}
        self.agents: list[AgentID] = []

        obs_size = len(DIRECTIONS) * view_range * N_CELL_FEATURES + 1
        self._observation_space = spaces.Box(
            0.0, 1.0, shape=(obs_size,), dtype=np.float32
        )
        self._action_space = spaces.Discrete(len(MOVES))

        self._rng = np.random.default_rng()

    def observation_space(self, agent: AgentID) -> spaces.Box:
        return self._observation_space

    def action_space(self, agent: AgentID) -> spaces.Discrete:
        return self._action_space

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._action_space.seed(seed)

        self.map = self.maps[self._rng.integers(len(self.maps))]
        self.walls = self.map.walls
        self.explored = np.zeros_like(self.walls, dtype=bool)
        self.explored[self.map.start] = True
        self.positions = np.tile(self.map.start, (len(self.possible_agents), 1))
        self.agents = self.possible_agents.copy()
        self.n_steps = 0

        return self._observations(), self._infos()

    def step(
        self, actions: dict[AgentID, int]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        if self.reward_mode == "mixed":
            rewards = self._mixed_reward(actions)
        else:
            rewards = self._grouped_reward(actions)
        self.n_steps += 1
        done = self.is_fully_explored()
        truncated = self.n_steps >= self.max_steps

        observations = self._observations()
        infos = self._infos()
        terminations = {a: done for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        if done or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def is_fully_explored(self) -> bool:
        return bool(self.explored[self.walls == 0].all())

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        chars = np.where(self.walls == 1, "#", np.where(self.explored, ".", " "))
        for row, col in self.positions:
            chars[row, col] = "D"
        return "\n".join("".join(line) for line in chars)

    def _grouped_reward(self, actions: dict[AgentID, int]) -> dict[AgentID, float]:
        total_new = sum(
            self._get_new_explored_cells_by_agent(
                self._agent_index[agent], actions.get(agent, 0)
            )
            for agent in self.agents
        )

        reward = total_new - self.step_penalty
        return {a: reward for a in self.agents}

    def _mixed_reward(self, actions: dict[AgentID, int]) -> dict[AgentID, float]:
        individual_new = {
            agent: self._get_new_explored_cells_by_agent(
                self._agent_index[agent], actions.get(agent, 0)
            )
            for agent in self.agents
        }
        total_new = sum(individual_new.values())

        alpha = self.reward_alpha
        return {
            agent: alpha * n_new + (1 - alpha) * total_new - self.step_penalty
            for agent, n_new in individual_new.items()
        }

    def _get_new_explored_cells_by_agent(self, i: int, action: int) -> int:
        row, col = self.positions[i] + MOVES[action]
        if self._is_free(row, col):
            self.positions[i] = row, col

        cell = (self.positions[i, 0], self.positions[i, 1])
        is_new = not self.explored[cell]
        self.explored[cell] = True
        return int(is_new)

    def _is_free(self, row: int, col: int) -> bool:
        height, width = self.walls.shape
        return 0 <= row < height and 0 <= col < width and self.walls[row, col] == 0

    def _observations(self) -> dict[AgentID, ObsType]:
        occupancy = np.zeros(self.walls.shape, dtype=np.int64)
        np.add.at(occupancy, (self.positions[:, 0], self.positions[:, 1]), 1)

        observations = {}
        for agent in self.agents:
            row, col = self.positions[self._agent_index[agent]]
            rays = np.zeros(
                (len(DIRECTIONS), self.view_range, N_CELL_FEATURES), dtype=np.float32
            )
            for d, (d_row, d_col) in enumerate(DIRECTIONS):
                for k in range(self.view_range):
                    r, c = row + d_row * (k + 1), col + d_col * (k + 1)
                    if not self._is_free(r, c):
                        rays[d, k:, 0] = 1.0
                        break
                    rays[d, k, 1] = self.explored[r, c]
                    rays[d, k, 2] = occupancy[r, c] > 0
            shares_cell = float(occupancy[row, col] > 1)
            observations[agent] = np.append(rays.ravel(), shares_cell).astype(
                np.float32
            )
        return observations

    def _infos(self) -> dict[AgentID, dict]:
        coverage = float(self.explored[self.walls == 0].mean())
        return {
            agent: {
                "position": tuple(self.positions[self._agent_index[agent]]),
                "coverage": coverage,
            }
            for agent in self.agents
        }
