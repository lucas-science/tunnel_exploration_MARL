from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

MAPS_DIR = Path(__file__).resolve().parent / "envs"


@dataclass(frozen=True)
class TunnelMap:
    walls: npt.NDArray[np.int8]
    start: tuple[int, int]

    @classmethod
    def from_csv(cls, path: str | Path) -> TunnelMap:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        start_col, start_row = int(rows[0][0]), int(rows[0][1])
        walls = np.array([[int(x) for x in row] for row in rows[1:]], dtype=np.int8)
        if walls[start_row, start_col] != 0:
            raise ValueError(f"Start cell {(start_col, start_row)} is a wall in {path}")
        return cls(walls=walls, start=(start_row, start_col))

    @classmethod
    def from_name(cls, name: str) -> TunnelMap:
        return cls.from_csv(MAPS_DIR / name)

    @property
    def shape(self) -> tuple[int, int]:
        return self.walls.shape[0], self.walls.shape[1]

    @property
    def n_tunnels(self) -> int:
        return int((self.walls == 0).sum())
