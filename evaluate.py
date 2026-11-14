from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import types
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmarl.experiment import Experiment
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from tqdm import tqdm

import benchmarl_task
from algo import ALGORITHMS
from report import write_report
from tunnel_env import GROUP, TunnelEnv
from tunnel_map import TunnelMap, map_family, map_set, resolve_maps

sys.modules["multi_agent_RL"] = types.ModuleType("multi_agent_RL")
sys.modules["multi_agent_RL.benchmarl_task"] = benchmarl_task

Frame = tuple[np.ndarray, np.ndarray, np.ndarray]
Actor = Callable[[TensorDictBase], TensorDictBase]
Key = tuple[str, str]

RUNS_DIR = Path(__file__).parent / "runs"
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


def gif_path(gif_dir: Path, map_name: str, mode: str) -> Path:
    return gif_dir / map_set(map_name) / map_family(map_name) / f"{Path(map_name).stem}_{mode}.gif"


def render_gif(frames: list[Frame], coverage: np.ndarray, title: str, output: Path, fps: int) -> None:
    discoveries = np.flatnonzero(np.diff(coverage) > 0)
    last = discoveries[-1] + 1 if len(discoveries) else 0
    end = min(last + 1 + GIF_TAIL, len(frames))
    output.parent.mkdir(parents=True, exist_ok=True)
    animate(frames[:end], coverage[:end], title).save(output, writer=PillowWriter(fps=fps))
    plt.close("all")


def animate(frames: list[Frame], coverage: np.ndarray, title: str) -> FuncAnimation:
    walls, _, first_positions = frames[0]
    colors = plt.get_cmap("tab10")(np.arange(len(first_positions)))
    trails = np.stack([positions for _, _, positions in frames])

    fig, (ax_map, ax_cov) = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle(title)
    image = ax_map.imshow(np.zeros(walls.shape), cmap=CELL_COLORS, vmin=0, vmax=2, interpolation="none")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    lines = [ax_map.plot([], [], color=color, alpha=0.5, lw=1.5)[0] for color in colors]
    drones = ax_map.scatter(first_positions[:, 1], first_positions[:, 0], c=colors, s=60,
                            edgecolors="black", zorder=3)

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


class Evaluator:
    modes: dict[str, ExplorationType]
    name: str
    gif_dir: Path
    config: dict
    device: torch.device

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *_) -> None:
        pass

    def actor(self, env: TunnelEnv, seed: int) -> Actor:
        raise NotImplementedError

    def make_env(self, tunnel_maps: list[TunnelMap], episodes: int, max_steps: int) -> TunnelEnv:
        env = TunnelEnv(
            tunnel_maps,
            num_envs=len(tunnel_maps) * episodes,
            n_agents=self.config["n_agents"],
            view_range=self.config["view_range"],
            max_steps=max_steps,
            hot_vector=self.config.get("hot_vector", False),
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
        patience: float,
        seed: int,
        animated: list[int],
        on_step: Callable[[], None],
    ) -> tuple[np.ndarray, np.ndarray, dict[int, list[Frame]]]:
        tunnel_maps = [TunnelMap.from_name(name) for name in maps]
        env = self.make_env(tunnel_maps, episodes, max_steps)
        env.set_seed(seed)
        torch.manual_seed(seed)
        map_index = env.env_index.cpu() % len(maps)
        reachable = torch.tensor([reachable_cells(m) for m in tunnel_maps])[map_index]
        limit = patience * torch.tensor([m.diameter for m in tunnel_maps], dtype=torch.float)[map_index]

        curve = torch.zeros(env.num_envs, max_steps + 1)
        lengths = torch.full((env.num_envs,), max_steps)
        active = torch.ones(env.num_envs, dtype=torch.bool)
        stalled = torch.zeros(env.num_envs)
        frames: dict[int, list[Frame]] = {m: [] for m in animated}

        def coverage() -> torch.Tensor:
            return ((~env.walls & env.explored).flatten(1).sum(-1).cpu() / reachable).clamp(max=1)

        with torch.no_grad(), set_exploration_type(self.modes[mode]):
            td = env.reset()
            actor = self.actor(env, seed)
            curve[:, 0] = coverage()
            for m in animated:
                frames[m].append(env.snapshot(m))
            for t in range(1, max_steps + 1):
                td = step_mdp(env.step(actor(td)))
                current = coverage()
                stalled = torch.where(current > curve[:, t - 1], 0, stalled + 1)
                curve[:, t] = torch.where(active, current, curve[:, t - 1])
                for m in animated:
                    if active[m]:
                        frames[m].append(env.snapshot(m))
                ended = active & ((current >= 1) | (stalled >= limit))
                lengths[ended] = t
                active &= ~ended
                on_step()
                if not active.any():
                    break

        curves = curve[:, : t + 1].view(episodes, len(maps), -1).transpose(0, 1).numpy()
        return curves, lengths.view(episodes, len(maps)).T.numpy(), frames


class PolicyEvaluator(Evaluator):
    modes = {"greedy": ExplorationType.DETERMINISTIC, "stochastic": ExplorationType.RANDOM}

    def __init__(self, run: Path, device: str) -> None:
        checkpoint = self.last_checkpoint(run)
        self.name = run.name
        self.gif_dir = run / "evaluation" / "animation"
        self._tmp = tempfile.TemporaryDirectory()
        self.experiment = Experiment.reload_from_file(
            str(checkpoint),
            experiment_patch={
                "save_folder": self._tmp.name,
                "loggers": [],
                "create_json": False,
                "evaluation_episodes": 1,
                "train_device": device,
                "sampling_device": device,
                "buffer_device": device,
                "restore_map_location": device,
            },
        )
        self.config = self.experiment.task.config
        self.device = next(self.experiment.policy.parameters()).device

    def __exit__(self, *_) -> None:
        self._tmp.cleanup()

    @staticmethod
    def last_checkpoint(run: Path) -> Path:
        checkpoints = sorted(run.glob("checkpoints/checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if not checkpoints:
            raise FileNotFoundError(f"Aucun checkpoint dans {run / 'checkpoints'}")
        return checkpoints[-1]

    def actor(self, env: TunnelEnv, seed: int) -> Actor:
        return self.experiment.policy


class AlgorithmEvaluator(Evaluator):
    modes = {"baseline": ExplorationType.DETERMINISTIC}

    def __init__(self, name: str, out: Path, n_agents: int, view_range: int, device: str) -> None:
        self.name = name
        self.gif_dir = out / "animation"
        self.config = {"n_agents": n_agents, "view_range": view_range}
        self.device = torch.device(device)

    def actor(self, env: TunnelEnv, seed: int) -> Actor:
        algorithm = ALGORITHMS[self.name](env, seed)
        algorithm.start()
        return lambda td: td.set((GROUP, "action"), algorithm.act(td))


@dataclass(frozen=True)
class Settings:
    maps: list[str]
    episodes: int
    max_steps: int
    patience: float
    seed: int
    fps: int
    device: str
    n_agents: int
    view_range: int
    out: Path


@dataclass(frozen=True)
class Job:
    run: Path | str
    mode: str
    maps: list[int]
    animated: list[int]


_steps = None


def init_worker(steps, threads: int) -> None:
    global _steps
    _steps = steps
    torch.set_num_threads(threads)


def count_step() -> None:
    with _steps.get_lock():
        _steps.value += 1


def make_evaluator(settings: Settings, run: Path | str) -> Evaluator:
    if isinstance(run, str):
        return AlgorithmEvaluator(run, settings.out, settings.n_agents, settings.view_range, settings.device)
    return PolicyEvaluator(run, settings.device)


def evaluate_job(settings: Settings, job: Job) -> tuple[str, Path, np.ndarray, np.ndarray, dict[int, list[Frame]]]:
    maps = [settings.maps[m] for m in job.maps]
    animated = [job.maps.index(m) for m in job.animated]
    with make_evaluator(settings, job.run) as evaluator:
        curves, lengths, frames = evaluator.run_episodes(
            maps, settings.episodes, job.mode, settings.max_steps, settings.patience, settings.seed,
            animated, count_step,
        )
        return evaluator.name, evaluator.gif_dir, curves, lengths, {job.maps[m]: f for m, f in frames.items()}


def modes_of(run: Path | str) -> list[str]:
    return list(AlgorithmEvaluator.modes if isinstance(run, str) else PolicyEvaluator.modes)


def animated_maps(maps: list[str], gif_sets: list[str]) -> list[int]:
    first = {(map_set(name), map_family(name)): m for m, name in reversed(list(enumerate(maps)))}
    return sorted(m for (s, _), m in first.items() if s in gif_sets)


def plan_jobs(runs: list[Path | str], n_maps: int, animated: list[int], workers: int) -> list[Job]:
    pairs = [(run, mode) for run in runs for mode in modes_of(run)]
    n_chunks = min(n_maps, -(-2 * workers // len(pairs)))
    chunks = [list(range(n_maps))[k::n_chunks] for k in range(n_chunks)]
    return [
        Job(run, mode, chunk, [m for m in animated if m in chunk] if run == runs[0] else [])
        for run, mode in pairs
        for chunk in chunks
    ]


def run_jobs(
    settings: Settings, jobs: list[Job], workers: int
) -> tuple[dict[Key, list[tuple[list[int], np.ndarray]]], dict[Key, np.ndarray], dict[Path | str, str]]:
    context = mp.get_context("spawn")
    steps = context.Value("l", 0)
    threads = max(1, (os.cpu_count() or 1) // workers)

    parts: dict[Key, list[tuple[list[int], np.ndarray]]] = {}
    lengths: dict[Key, np.ndarray] = {}
    names: dict[Path | str, str] = {}
    gifs: list[Future] = []
    with ProcessPoolExecutor(workers, mp_context=context, initializer=init_worker,
                             initargs=(steps, threads)) as pool, \
            tqdm(unit="step", dynamic_ncols=True) as progress:
        pending = {pool.submit(evaluate_job, settings, job): job for job in jobs}
        while pending or not all(g.done() for g in gifs):
            done, _ = wait(list(pending) or gifs, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done & pending.keys():
                job = pending.pop(future)
                name, gif_dir, curve, length, frames = future.result()
                key = (name, job.mode)
                names[job.run] = name
                parts.setdefault(key, []).append((job.maps, curve))
                lengths.setdefault(key, np.zeros((len(settings.maps), settings.episodes), int))[job.maps] = length
                for m, map_frames in frames.items():
                    map_name = settings.maps[m]
                    coverage = curve[job.maps.index(m), 0]
                    title = f"{name} — {map_name} ({job.mode})"
                    output = gif_path(gif_dir, map_name, job.mode)
                    gifs.append(pool.submit(render_gif, map_frames, coverage, title, output, settings.fps))
            progress.n = steps.value
            progress.set_postfix_str(f"{len(jobs) - len(pending)}/{len(jobs)} évaluations, "
                                     f"{sum(g.done() for g in gifs)}/{len(gifs)} GIFs", refresh=False)
            progress.refresh()
        for g in gifs:
            g.result()
    return parts, lengths, names


def merge_chunks(parts: dict[Key, list[tuple[list[int], np.ndarray]]], n_maps: int, episodes: int) -> dict[Key, np.ndarray]:
    n_steps = max(curve.shape[-1] for chunks in parts.values() for _, curve in chunks)
    curves = {}
    for key, chunks in parts.items():
        curves[key] = np.zeros((n_maps, episodes, n_steps))
        for maps, curve in chunks:
            curves[key][maps] = np.pad(curve, ((0, 0), (0, 0), (0, n_steps - curve.shape[-1])), mode="edge")
    return curves


def find_runs(folder: Path) -> list[Path]:
    if (folder / "checkpoints").is_dir():
        return [folder]
    return sorted(p for p in folder.iterdir() if (p / "checkpoints").is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help=f"dossier d'expérience ou d'un run, ou algorithme : {', '.join(ALGORITHMS)}")
    parser.add_argument("--maps", nargs="+", default=["small_maps/val", "big_maps/val"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=10000, help="plafond de sécurité")
    parser.add_argument("--patience", type=float, default=2.0,
                        help="fin d'épisode après patience × diamètre de la carte pas sans découverte")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-maps", nargs="+", default=["small_maps"], help="jeux de cartes animés")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--n-agents", type=int, default=4, help="algorithme uniquement")
    parser.add_argument("--view-range", type=int, default=3, help="algorithme uniquement")
    parser.add_argument("--experiment", help="algorithme uniquement : sous-dossier de runs/ (défaut Baseline_<algo>)")
    args = parser.parse_args()
    try:
        args.maps = resolve_maps([name.lstrip("/") for name in args.maps])
    except FileNotFoundError as error:
        parser.error(str(error))
    if args.target in ALGORITHMS:
        args.folder = RUNS_DIR / (args.experiment or f"Baseline_{args.target}")
        args.runs = [args.target]
        return args
    args.folder = Path(args.target)
    if not args.folder.is_dir():
        parser.error(f"{args.target} n'est ni un dossier ni un algorithme ({', '.join(ALGORITHMS)})")
    args.runs = find_runs(args.folder)
    if not args.runs:
        parser.error(f"aucun run avec checkpoints/ dans {args.folder}")
    return args


def main() -> None:
    args = parse_args()
    out = args.folder / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings(args.maps, args.episodes, args.max_steps, args.patience, args.seed, args.fps,
                        args.device, args.n_agents, args.view_range, out)
    animated = [] if args.no_gif else animated_maps(args.maps, args.gif_maps)
    jobs = plan_jobs(args.runs, len(args.maps), animated, args.workers)
    workers = max(1, min(args.workers, len(jobs)))

    parts, lengths, names = run_jobs(settings, jobs, workers)
    curves = merge_chunks(parts, len(args.maps), args.episodes)
    write_report(out, curves, lengths, [names[run] for run in args.runs], args.maps)

    print((out / "summary.md").read_text())
    print(f"Résultats : {out}")


if __name__ == "__main__":
    main()
