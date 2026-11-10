from __future__ import annotations

import argparse
import random
import sys
import tempfile
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmarl.experiment import Experiment
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from torchrl.envs.utils import ExplorationType, set_exploration_type

import benchmarl_task
from benchmarl_task import unwrap
from tunnel_env import GROUP
from tunnel_map import TunnelMap, map_names

sys.modules["multi_agent_RL"] = types.ModuleType("multi_agent_RL")
sys.modules["multi_agent_RL.benchmarl_task"] = benchmarl_task

Frame = tuple[np.ndarray, np.ndarray, np.ndarray]
CELL_COLORS = ListedColormap(["#d9d9d9", "#9fd8a0", "#2b2b2b"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="dossier du run ou fichier .pt")
    parser.add_argument("--map", choices=map_names())
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def find_checkpoint(path: Path) -> Path:
    if path.suffix == ".pt":
        return path
    checkpoints = sorted(
        path.glob("checkpoints/checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"Aucun checkpoint dans {path / 'checkpoints'}")
    return checkpoints[-1]


def run_episode(
    experiment: Experiment, map_name: str | None, seed: int, stochastic: bool
) -> tuple[list[Frame], np.ndarray, str]:
    env = experiment.test_env
    tunnel_env = unwrap(env)
    if map_name is not None:
        tunnel_env.set_maps(TunnelMap.from_name(map_name))
    env.set_seed(seed)
    torch.manual_seed(seed)

    frames: list[Frame] = []
    exploration = (
        ExplorationType.RANDOM if stochastic else ExplorationType.DETERMINISTIC
    )
    with torch.no_grad(), set_exploration_type(exploration):
        td = env.reset()
        frames.append(tunnel_env.snapshot())
        rollout = env.rollout(
            tunnel_env.max_steps,
            policy=experiment.policy,
            tensordict=td,
            auto_reset=False,
            auto_cast_to_device=True,
            break_when_any_done=True,
            callback=lambda *_: frames.append(tunnel_env.snapshot()),
        )
    if len(frames) == rollout.batch_size[-1]:
        frames.append(tunnel_env.snapshot())

    coverage = rollout.get(("next", GROUP, "info", "coverage"))[0, :, 0].cpu().numpy()
    if map_name is None:
        map_name = experiment.task.config["maps"][int(tunnel_env.map_ids[0])]
    return frames, np.concatenate([[coverage[0]], coverage]), str(map_name)


def animate(frames: list[Frame], coverage: np.ndarray, title: str) -> FuncAnimation:
    walls, _, first_positions = frames[0]
    n_agents = len(first_positions)
    colors = plt.get_cmap("tab10")(np.arange(n_agents))
    trails = np.stack([positions for _, _, positions in frames])

    fig, (ax_map, ax_cov) = plt.subplots(
        1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    fig.suptitle(title)
    image = ax_map.imshow(
        np.zeros(walls.shape), cmap=CELL_COLORS, vmin=0, vmax=2, interpolation="none"
    )
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    lines = [
        ax_map.plot([], [], color=colors[a], alpha=0.5, lw=1.5)[0]
        for a in range(n_agents)
    ]
    drones = ax_map.scatter(
        first_positions[:, 1],
        first_positions[:, 0],
        c=colors,
        s=60,
        edgecolors="black",
        zorder=3,
    )

    steps = np.arange(len(coverage))
    ax_cov.plot(steps, coverage * 100, color="#2b7bb9")
    ax_cov.set_xlim(0, max(len(coverage) - 1, 1))
    ax_cov.set_ylim(0, 100)
    ax_cov.set_xlabel("step")
    ax_cov.set_ylabel("couverture (%)")
    ax_cov.grid(alpha=0.3)
    marker = ax_cov.axvline(0, color="black", lw=1)

    def update(t: int):
        walls, explored, positions = frames[t]
        image.set_data(np.where(walls, 2, np.where(explored, 1, 0)))
        drones.set_offsets(positions[:, ::-1])
        for a, line in enumerate(lines):
            line.set_data(trails[: t + 1, a, 1], trails[: t + 1, a, 0])
        marker.set_xdata([t, t])
        ax_map.set_title(f"step {t}  —  couverture {coverage[t]:.0%}")
        return [image, drones, marker, *lines]

    return FuncAnimation(fig, update, frames=len(frames), blit=False)


def main() -> None:
    args = parse_args()
    if not args.live:
        plt.switch_backend("Agg")
    checkpoint = find_checkpoint(args.run)
    seed = args.seed if args.seed is not None else random.randrange(2**31)

    with tempfile.TemporaryDirectory() as tmp:
        experiment = Experiment.reload_from_file(
            str(checkpoint),
            experiment_patch={
                "save_folder": tmp,
                "loggers": [],
                "create_json": False,
                "evaluation_episodes": 1,
            },
        )
        frames, coverage, map_name = run_episode(
            experiment, args.map, seed, args.stochastic
        )

    print(
        f"{map_name} (seed {seed}) : {len(frames) - 1} steps, "
        f"couverture finale {coverage[-1]:.1%}"
    )
    animation = animate(
        frames, coverage, f"{checkpoint.parent.parent.name} — {map_name}"
    )
    if args.live:
        plt.show()
        return
    output = (
        args.output or checkpoint.parent.parent / f"policy_{Path(map_name).stem}.gif"
    )
    animation.save(output, writer=PillowWriter(fps=args.fps))
    print(f"GIF : {output}")


if __name__ == "__main__":
    main()
