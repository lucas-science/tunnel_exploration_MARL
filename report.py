from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tunnel_map import map_family, map_set

Curves = dict[tuple[str, str], np.ndarray]
ModeCurves = dict[str, np.ndarray]
ModeScores = dict[str, dict[str, np.ndarray]]

MODE_COLORS = {"greedy": "#4C72B0", "stochastic": "#DD8452", "baseline": "#55A868"}
MODE_LABELS = {"greedy": "Greedy", "stochastic": "Stochastic", "baseline": "Baseline"}
FAMILY_LABELS = {"corner_start_deadend": "corner", "braided_maze": "braided", "grid_city": "grid"}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.linewidth": 0.8,
    "grid.color": "0.9",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
}
ERROR_KW = {"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "0.15"}
SPEED_THRESHOLD = 0.9
T_95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086]

EPISODE_METRICS = {"coverage": "final_coverage", "solved_rate": "solved", "reached_90_rate": "reached_90",
                   "steps_to_90": "steps_to_90"}
METRIC_NAMES = ("Couverture finale", "Épisodes à 100 %", "Épisodes à 90 %", "Pas pour 90 %")
METRIC_FORMATS = {"coverage": ".1%", "solved_rate": ".1%", "reached_90_rate": ".1%", "steps_to_90": ".0f"}
SUMMARY_NOTES = [
    "Valeurs : moyenne ± écart-type entre runs (un score par run : moyenne par carte, puis sur les cartes du groupe).",
    "Un épisode s'arrête quand toutes les cases atteignables sont explorées, ou quand plus aucune case "
    "n'est découverte pendant patience × diamètre de la carte pas.",
    f"Pas pour 90 % : premier pas où {SPEED_THRESHOLD:.0%} des cases atteignables sont explorées, "
    "moyenne sur les seuls épisodes qui y arrivent (leur part est dans « Épisodes à 90 % »).",
]


def unique(values) -> list:
    return list(dict.fromkeys(values))


def of_set(rows: list[dict], map_set_name: str) -> list[dict]:
    return [r for r in rows if r["maps"] == map_set_name]


def quiet_nanmean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis)


def mean_ci(values: np.ndarray, bounded: bool = True) -> tuple[np.ndarray, ...]:
    if values.ndim == 1:
        values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return (np.nan,) * 4
    mean = values.mean(0)
    if n < 2:
        nan = np.full_like(mean, np.nan, dtype=float)
        return mean, nan, nan, nan
    std = values.std(0, ddof=1)
    t = T_95[n - 2] if n - 2 < len(T_95) else 1.96
    half = t * std / np.sqrt(n)
    low, high = mean - half, mean + half
    if bounded:
        low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
    return mean, std, low, high


def map_groups(maps: list[str]) -> dict[str, list[int]]:
    families = sorted({map_family(m) for m in maps})
    return {"all": list(range(len(maps)))} | {
        f: [i for i, m in enumerate(maps) if map_family(m) == f] for f in families
    }


def family_groups(maps: list[str]) -> dict[str, list[int]]:
    return {g: idx for g, idx in map_groups(maps).items() if g != "all"}


def steps_to_threshold(curves: np.ndarray) -> np.ndarray:
    reached = curves >= SPEED_THRESHOLD - 1e-9
    return np.where(reached.any(-1), reached.argmax(-1), np.nan)


def split_by_set(curves: Curves, lengths: Curves, maps: list[str]) -> dict[str, tuple[Curves, Curves, list[str]]]:
    by_set = {}
    for s in unique(map_set(m) for m in maps):
        idx = [i for i, m in enumerate(maps) if map_set(m) == s]
        n_steps = max(int(length[idx].max()) for length in lengths.values()) + 1
        by_set[s] = ({key: curve[idx, :, :n_steps] for key, curve in curves.items()},
                     {key: length[idx] for key, length in lengths.items()},
                     [maps[i] for i in idx])
    return by_set


def episode_rows(curves: Curves, lengths: Curves, maps: list[str]) -> list[dict]:
    rows = []
    for (run, mode), curve in curves.items():
        for m, name in enumerate(maps):
            for e, episode in enumerate(curve[m]):
                full = np.flatnonzero(episode >= 1)
                steps_to_90 = steps_to_threshold(episode)
                rows.append({
                    "run": run, "mode": mode, "map": name, "family": map_family(name),
                    "episode": e, "length": lengths[run, mode][m, e], "final_coverage": episode[-1],
                    "solved": int(len(full) > 0),
                    "steps_to_full": full[0] if len(full) else "",
                    "reached_90": int(not np.isnan(steps_to_90)),
                    "steps_to_90": steps_to_90,
                })
    return rows


def per_map_rows(curves: Curves, maps: list[str]) -> list[dict]:
    return [
        {"run": run, "mode": mode, "map": name, "family": map_family(name),
         "step": step, "coverage": value}
        for (run, mode), curve in curves.items()
        for m, name in enumerate(maps)
        for step, value in enumerate(curve[m].mean(0))
    ]


def curve_rows(curves: Curves, runs: list[str], maps: list[str]) -> list[dict]:
    rows = []
    for mode in unique(mode for _, mode in curves):
        per_map = np.stack([curves[run, mode] for run in runs]).mean(2)
        for group, idx in map_groups(maps).items():
            for step, (mean, _, low, high) in enumerate(zip(*mean_ci(per_map[:, idx].mean(1)))):
                rows.append({"mode": mode, "group": group, "step": step,
                             "mean": mean, "ci_low": low, "ci_high": high})
    return rows


def index_rows(rows: list[dict], column: str) -> dict[tuple[str, str, str], list[float]]:
    index: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        index.setdefault((r["mode"], r["run"], r["map"]), []).append(float(r[column]))
    return index


def map_scores(rows: list[dict], runs: list[str], maps: list[str], mode: str) -> dict[str, np.ndarray]:
    scores = {}
    for key, column in EPISODE_METRICS.items():
        index = index_rows(rows, column)
        scores[key] = np.array([[quiet_nanmean(index[mode, run, name]) for name in maps] for run in runs])
    return scores


def summary_rows(episodes: list[dict]) -> list[dict]:
    rows = []
    for s in unique(r["maps"] for r in episodes):
        set_episodes = of_set(episodes, s)
        runs = unique(r["run"] for r in set_episodes)
        maps = unique(r["map"] for r in set_episodes)
        for mode in unique(r["mode"] for r in set_episodes):
            scores = map_scores(set_episodes, runs, maps, mode)
            for group, idx in map_groups(maps).items():
                row = {"maps": s, "mode": mode, "group": group, "n_runs": len(runs), "n_maps": len(idx)}
                for key, per_map in scores.items():
                    per_run = quiet_nanmean(per_map[:, idx], 1)
                    mean, std, low, high = mean_ci(per_run, bounded=key != "steps_to_90")
                    row |= {f"{key}_mean": mean, f"{key}_std": std, f"{key}_ci_low": low, f"{key}_ci_high": high}
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt_std(mean: float, std: float, fmt: str) -> str:
    if np.isnan(mean):
        return "—"
    if np.isnan(std):
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def metric_cells(row: dict) -> list[str]:
    return [fmt_std(float(row[f"{key}_mean"]), float(row[f"{key}_std"]), fmt) for key, fmt in METRIC_FORMATS.items()]


def table_header(columns: list[str]) -> list[str]:
    return ["| " + " | ".join(columns) + " |", "| --- |" + " --- |" * (len(columns) - 1)]


def overview_lines(rows: list[dict], sets: list[str]) -> list[str]:
    lines = table_header(["Mode", *(f"{name} {s}" for s in sets for name in METRIC_NAMES)])
    for mode in unique(r["mode"] for r in rows):
        cells = [cell for s in sets for r in of_set(rows, s)
                 if r["mode"] == mode and r["group"] == "all" for cell in metric_cells(r)]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return lines


def detail_lines(rows: list[dict]) -> list[str]:
    lines = table_header(["Mode", "Groupe", "Runs", "Cartes", *METRIC_NAMES])
    for r in rows:
        cells = [r["mode"], r["group"], r["n_runs"], r["n_maps"], *metric_cells(r)]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    return lines


def write_markdown(path: Path, rows: list[dict]) -> None:
    sets = unique(r["maps"] for r in rows)
    lines = []
    if len(sets) > 1:
        lines += ["## Vue d'ensemble", "", *overview_lines(rows, sets), ""]
    for s in sets:
        lines += [f"## {s}", "", *detail_lines(of_set(rows, s)), ""]
    for note in SUMMARY_NOTES:
        lines += [note, ""]
    path.write_text("\n".join(lines))


def write_summary(out: Path, episodes: list[dict]) -> None:
    summary = summary_rows(episodes)
    write_csv(out / "summary.csv", summary)
    write_markdown(out / "summary.md", summary)


def family_label(group: str) -> str:
    return "All" if group == "all" else FAMILY_LABELS.get(group, group)


def top_legend(fig: plt.Figure, modes: list[str]) -> None:
    handles = [Line2D([], [], color=MODE_COLORS[m], lw=2, label=MODE_LABELS[m]) for m in modes]
    fig.legend(handles=handles, loc="upper center", ncol=len(modes), bbox_to_anchor=(0.5, 1.0))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def coverage_axis(ax: plt.Axes, n_steps: int) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, n_steps - 1)
    ax.set_ylim(0, 100)


def draw_coverage(ax: plt.Axes, curves: ModeCurves, idx: list[int]) -> None:
    for mode, per_map in curves.items():
        per_run = per_map[:, idx].mean(1) * 100
        mean = per_run.mean(0)
        steps = np.arange(1, len(mean))
        if len(per_run) > 1:
            std = per_run.std(0, ddof=1)
            ax.fill_between(steps, (mean - std)[1:], (mean + std)[1:], color=MODE_COLORS[mode], alpha=0.2, lw=0)
        ax.plot(steps, mean[1:], color=MODE_COLORS[mode], lw=1.6)


def n_steps_of(curves: ModeCurves) -> int:
    return next(iter(curves.values())).shape[-1]


def plot_coverage(path: Path, curves: ModeCurves, maps: list[str]) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        draw_coverage(ax, curves, list(range(len(maps))))
        coverage_axis(ax, n_steps_of(curves))
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Coverage (%)")
        top_legend(fig, list(curves))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save(fig, path)


def plot_coverage_by_family(path: Path, curves: ModeCurves, maps: list[str]) -> None:
    groups = family_groups(maps)
    n_cols = min(4, len(groups))
    n_rows = -(-len(groups) // n_cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols, 1.6 * n_rows + 0.35),
                                 sharex=True, sharey=True, squeeze=False)
        panels = list(axes.flat)[:len(groups)]
        for ax, (group, idx) in zip(panels, groups.items()):
            draw_coverage(ax, curves, idx)
            coverage_axis(ax, n_steps_of(curves))
            ax.set_title(family_label(group))
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("Coverage (%)")
        for ax in panels[-n_cols:]:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("Steps")
        top_legend(fig, list(curves))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
        save(fig, path)


def plot_final_coverage(path: Path, scores: ModeScores, maps: list[str]) -> None:
    groups = family_groups(maps)
    modes = list(scores)
    x = np.arange(len(groups))
    width = 0.8 / len(modes)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for k, mode in enumerate(modes):
            per_run = np.stack([scores[mode]["coverage"][:, idx].mean(1) * 100 for idx in groups.values()])
            std = per_run.std(1, ddof=1) if per_run.shape[1] > 1 else None
            ax.bar(x + (k - (len(modes) - 1) / 2) * width, per_run.mean(1), width, yerr=std,
                   color=MODE_COLORS[mode], error_kw=ERROR_KW)
        ax.set_xticks(x, [family_label(g) for g in groups], rotation=30, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Final coverage (%)")
        top_legend(fig, modes)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save(fig, path)


def write_plots(out: Path, episodes: list[dict], per_map: list[dict]) -> None:
    for s in unique(r["maps"] for r in episodes):
        set_episodes = of_set(episodes, s)
        runs = unique(r["run"] for r in set_episodes)
        maps = unique(r["map"] for r in set_episodes)
        modes = unique(r["mode"] for r in set_episodes)
        coverage = index_rows(of_set(per_map, s), "coverage")
        curves = {mode: np.array([[coverage[mode, run, name] for name in maps] for run in runs]) for mode in modes}
        scores = {mode: map_scores(set_episodes, runs, maps, mode) for mode in modes}

        plots = out / "plots" / s
        plots.mkdir(parents=True, exist_ok=True)
        plot_coverage(plots / "coverage", curves, maps)
        plot_coverage_by_family(plots / "coverage_by_family", curves, maps)
        plot_final_coverage(plots / "final_coverage", scores, maps)


def write_report(out: Path, curves: Curves, lengths: Curves, runs: list[str], maps: list[str]) -> None:
    episodes, per_map, coverage_curve = [], [], []
    for s, (set_curves, set_lengths, set_maps) in split_by_set(curves, lengths, maps).items():
        episodes += [{"maps": s, **r} for r in episode_rows(set_curves, set_lengths, set_maps)]
        per_map += [{"maps": s, **r} for r in per_map_rows(set_curves, set_maps)]
        coverage_curve += [{"maps": s, **r} for r in curve_rows(set_curves, runs, set_maps)]

    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "coverage_per_map.csv", per_map)
    write_csv(out / "coverage_curve.csv", coverage_curve)
    write_summary(out, episodes)
    write_plots(out, episodes, per_map)


if __name__ == "__main__":
    import sys

    for folder in map(Path, sys.argv[1:]):
        episodes = read_csv(folder / "episodes.csv")
        write_summary(folder, episodes)
        write_plots(folder, episodes, read_csv(folder / "coverage_per_map.csv"))
        print(folder)
