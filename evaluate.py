from __future__ import annotations

import argparse
import sys
import tempfile
import types
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmarl.experiment import Experiment
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from tqdm import tqdm

import benchmarl_task
from report import write_report
from tunnel_env import TunnelEnv
from tunnel_map import TunnelMap, map_family, resolve_maps

sys.modules["multi_agent_RL"] = types.ModuleType("multi_agent_RL")
sys.modules["multi_agent_RL.benchmarl_task"] = benchmarl_task

Frame = tuple[np.ndarray, np.ndarray, np.ndarray]

MODES = {"greedy": ExplorationType.DETERMINISTIC, "stochastic": ExplorationType.RANDOM}
CELL_COLORS = ListedColormap(["#d9d9d9", "#9fd8a0", "#2b2b2b"])
GIF_TAIL = 10


def reachable_cells(tunnel_map: TunnelMap) -> int:
    free = tunnel_map.walls == 0
    seen = {tunnel_map.start}
    queue = deque([tunnel_map.start])
    while queue:
        r, c = queue.popleft()
        for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            inside = 0 <= cell[0] < free.shape[0] and 0 <= cell[1] < free.shape[1]
            if inside and free[cell] and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen)


class PolicyEvaluator:
    def __init__(self, run: Path, fps: int = 5) -> None:
        self.checkpoint = self.find_checkpoint(run)
        self.run_dir = self.checkpoint.parent.parent
        self.fps = fps
        self._tmp = tempfile.TemporaryDirectory()
        self.experiment = Experiment.reload_from_file(
            str(self.checkpoint),
            experiment_patch={
                "save_folder": self._tmp.name,
                "loggers": [],
                "create_json": False,
                "evaluation_episodes": 1,
            },
        )
        self.device = next(self.experiment.policy.parameters()).device

    def __enter__(self) -> PolicyEvaluator:
        return self

    def __exit__(self, *_) -> None:
        self._tmp.cleanup()

    @staticmethod
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

    def make_env(self, tunnel_maps: list[TunnelMap], episodes: int, max_steps: int) -> TunnelEnv:
        config = self.experiment.task.config
        env = TunnelEnv(
            tunnel_maps,
            num_envs=len(tunnel_maps) * episodes,
            n_agents=config["n_agents"],
            view_range=config["view_range"],
            max_steps=max_steps,
            hot_vector=config.get("hot_vector", False),
            device=self.device,
        )
        env.fixed_maps = True
        env.map_diameters.fill_(max_steps + 1)
        return env

    def run_episodes(
        self,
        maps: list[str],
        episodes: int,
        mode: str,
        max_steps: int,
        seed: int,
        animated: list[int],
        progress: tqdm,
    ) -> tuple[np.ndarray, dict[int, list[Frame]]]:
        tunnel_maps = [TunnelMap.from_name(name) for name in maps]
        env = self.make_env(tunnel_maps, episodes, max_steps)
        env.set_seed(seed)
        torch.manual_seed(seed)
        reachable = torch.tensor(
            [reachable_cells(m) for m in tunnel_maps], device=self.device
        )[env.env_index % len(maps)]

        curve = torch.zeros(env.num_envs, max_steps + 1)
        frames: dict[int, list[Frame]] = {m: [] for m in animated}

        def record(t: int) -> None:
            explored = (~env.walls & env.explored).flatten(1).sum(-1)
            curve[:, t] = (explored / reachable).cpu()
            for m in animated:
                frames[m].append(env.snapshot(m))

        with torch.no_grad(), set_exploration_type(MODES[mode]):
            td = env.reset()
            record(0)
            for t in range(1, max_steps + 1):
                td = step_mdp(env.step(self.experiment.policy(td)))
                record(t)
                progress.set_postfix_str(f"couverture {curve[:, t].mean():.1%}", refresh=False)
                progress.update()
        return curve.view(episodes, len(maps), -1).transpose(0, 1).numpy(), frames

    @staticmethod
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

        ax_cov.plot(coverage * 100, color="#2b7bb9")
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

    def render(
        self, frames: list[Frame], coverage: np.ndarray, map_name: str, mode: str
    ) -> Path:
        discoveries = np.flatnonzero(np.diff(coverage) > 0)
        last = discoveries[-1] + 1 if len(discoveries) else 0
        end = min(last + 1 + GIF_TAIL, len(frames))
        animation = self.animate(
            frames[:end], coverage[:end], f"{self.run_dir.name} — {map_name} ({mode})"
        )
        folder = self.run_dir / "evaluation" / "animation" / map_family(map_name)
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{Path(map_name).stem}_{mode}.gif"
        animation.save(output, writer=PillowWriter(fps=self.fps))
        plt.close("all")
        return output

    def evaluate(
        self,
        maps: list[str],
        episodes: int,
        max_steps: int,
        seed: int,
        gif: bool,
        progress: tqdm,
        label: str,
    ) -> dict[str, np.ndarray]:
        first_of_family = {map_family(name): m for m, name in reversed(list(enumerate(maps)))}
        animated = sorted(first_of_family.values()) if gif else []
        results = {}
        for mode in MODES:
            progress.set_description(f"{label} {mode}")
            curves, frames = self.run_episodes(
                maps, episodes, mode, max_steps, seed, animated, progress
            )
            results[mode] = curves
            progress.write(f"{self.run_dir.name} {mode} : couverture finale {curves[..., -1].mean():.1%}")
            for m in animated:
                progress.set_postfix_str(f"GIF {Path(maps[m]).stem}")
                self.render(frames[m], curves[m, 0], maps[m], mode)
        return results


def find_runs(folder: Path) -> list[Path]:
    if (folder / "checkpoints").is_dir():
        return [folder]
    return sorted(p for p in folder.iterdir() if (p / "checkpoints").is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="dossier d'expérience ou d'un run")
    parser.add_argument("--maps", nargs="+", default=["small_maps/val"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    try:
        args.maps = resolve_maps([name.lstrip("/") for name in args.maps])
    except FileNotFoundError as error:
        parser.error(str(error))
    args.runs = find_runs(args.folder)
    if not args.runs:
        parser.error(f"aucun run avec checkpoints/ dans {args.folder}")
    return args


def main() -> None:
    args = parse_args()
    out = args.folder / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    curves: dict[tuple[str, str], np.ndarray] = {}
    total = len(args.runs) * len(MODES) * args.max_steps
    with tqdm(total=total, unit="step", dynamic_ncols=True) as progress:
        for i, run in enumerate(args.runs, 1):
            progress.set_description(f"run {i}/{len(args.runs)} chargement")
            with PolicyEvaluator(run, fps=args.fps) as evaluator:
                results = evaluator.evaluate(
                    args.maps, args.episodes, args.max_steps, args.seed, not args.no_gif,
                    progress, f"run {i}/{len(args.runs)}",
                )
            curves |= {(run.name, mode): curve for mode, curve in results.items()}

    write_report(out, curves, [run.name for run in args.runs], args.maps)

    print((out / "summary.md").read_text())
    print(f"Résultats : {out}")


if __name__ == "__main__":
    main()
