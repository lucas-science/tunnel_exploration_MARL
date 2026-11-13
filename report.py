from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tunnel_map import map_family

Curves = dict[tuple[str, str], np.ndarray]

MODE_COLORS = {"greedy": "#2a78d6", "stochastic": "#eb6834"}
MODE_LABELS = {"greedy": "greedy", "stochastic": "stochastique"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "savefig.dpi": 200,
}
T_95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086]


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    n = len(values)
    mean = values.mean(0)
    if n < 2:
        nan = np.full_like(mean, np.nan, dtype=float)
        return mean, nan, nan, nan
    std = values.std(0, ddof=1)
    half = (T_95[n - 2] if n - 2 < len(T_95) else 1.96) * std / np.sqrt(n)
    return mean, std, np.clip(mean - half, 0, 1), np.clip(mean + half, 0, 1)


def modes_of(curves: Curves) -> list[str]:
    return list(dict.fromkeys(mode for _, mode in curves))


def map_groups(maps: list[str]) -> dict[str, list[int]]:
    families = sorted({map_family(m) for m in maps})
    return {"all": list(range(len(maps)))} | {
        f: [i for i, m in enumerate(maps) if map_family(m) == f] for f in families
    }


def run_curves(curves: Curves, runs: list[str], mode: str) -> np.ndarray:
    return np.stack([curves[run, mode] for run in runs]).mean(2)


def episode_rows(curves: Curves, maps: list[str]) -> list[dict]:
    rows = []
    for (run, mode), curve in curves.items():
        for m, name in enumerate(maps):
            for e, episode in enumerate(curve[m]):
                full = np.flatnonzero(episode >= 1)
                rows.append({
                    "run": run, "mode": mode, "map": name, "family": map_family(name),
                    "episode": e, "final_coverage": episode[-1],
                    "solved": int(len(full) > 0),
                    "steps_to_full": full[0] if len(full) else "",
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


def summary_and_curve_rows(
    curves: Curves, runs: list[str], maps: list[str]
) -> tuple[list[dict], list[dict]]:
    summary, curve_rows = [], []
    for mode in modes_of(curves):
        per_map = run_curves(curves, runs, mode)
        solved_per_map = (np.stack([curves[run, mode] for run in runs])[..., -1] >= 1).mean(2)
        for group, idx in map_groups(maps).items():
            group_curves = per_map[:, idx].mean(1)
            cov_mean, cov_std, cov_low, cov_high = mean_ci(group_curves[:, -1])
            sol_mean, _, sol_low, sol_high = mean_ci(solved_per_map[:, idx].mean(1))
            summary.append({
                "mode": mode, "group": group, "n_runs": len(runs), "n_maps": len(idx),
                "coverage_mean": cov_mean, "coverage_std": cov_std,
                "coverage_ci_low": cov_low, "coverage_ci_high": cov_high,
                "solved_rate_mean": sol_mean,
                "solved_rate_ci_low": sol_low, "solved_rate_ci_high": sol_high,
            })
            for step, (mean, _, low, high) in enumerate(zip(*mean_ci(group_curves))):
                curve_rows.append({
                    "mode": mode, "group": group, "step": step,
                    "mean": mean, "ci_low": low, "ci_high": high,
                })
    return summary, curve_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_ci(mean: float, low: float, high: float) -> str:
    if np.isnan(low):
        return f"{mean:.1%}"
    return f"{mean:.1%} [{low:.1%}, {high:.1%}]"


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Mode | Groupe | Runs | Cartes | Couverture finale (IC 95 %) | Écart-type | Épisodes à 100 % (IC 95 %) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        std = "" if np.isnan(r["coverage_std"]) else f"{r['coverage_std']:.1%}"
        coverage = fmt_ci(r["coverage_mean"], r["coverage_ci_low"], r["coverage_ci_high"])
        solved = fmt_ci(r["solved_rate_mean"], r["solved_rate_ci_low"], r["solved_rate_ci_high"])
        lines.append(
            f"| {r['mode']} | {r['group']} | {r['n_runs']} | {r['n_maps']} "
            f"| {coverage} | {std} | {solved} |"
        )
    path.write_text("\n".join(lines) + "\n")


def legend_handles(modes: list[str]) -> list[Line2D]:
    return [
        Line2D([], [], color=MODE_COLORS[m], lw=2, marker="o", ms=5, label=MODE_LABELS[m])
        for m in modes
    ]


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def header(fig: plt.Figure, title: str, text: str, modes: list[str] | None = None) -> None:
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.18 / height, title, ha="left", va="top", fontsize=13, fontweight="semibold")
    fig.text(0.01, 1 - 0.48 / height, text, ha="left", va="top", color=INK_SECONDARY, fontsize=9.5)
    if modes:
        fig.legend(handles=legend_handles(modes), loc="upper left", ncol=len(modes),
                   bbox_to_anchor=(0.0, 1 - 0.68 / height), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - (1.0 if modes else 0.75) / height))


def runs_text(runs: list[str], episodes: int) -> str:
    n = len(runs)
    ci = "IC 95 % entre runs" if n > 1 else "un seul run, pas d'IC"
    return f"Moyenne sur {n} run{'s' * (n > 1)} ({episodes} épisodes par carte) · {ci}"


def coverage_axis(ax: plt.Axes, n_steps: int) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, n_steps - 1)
    ticks = [t for t in (1, 10, 100, 1000) if t < n_steps]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])


def draw_coverage(ax: plt.Axes, curves: Curves, runs: list[str], idx: list[int], labels: bool) -> None:
    ends = []
    for mode in modes_of(curves):
        mean, _, low, high = (v * 100 for v in mean_ci(run_curves(curves, runs, mode)[:, idx].mean(1)))
        steps = np.arange(len(mean))
        color = MODE_COLORS[mode]
        if not np.isnan(low).all():
            ax.fill_between(steps[1:], low[1:], high[1:], color=color, alpha=0.18, lw=0)
        ax.plot(steps[1:], mean[1:], color=color, lw=2, solid_capstyle="round")
        ends.append((mean[-1], mode))
    if labels:
        ends.sort()
        previous = -10.0
        for value, mode in ends:
            y = max(value, previous + 7)
            previous = y
            ax.annotate(
                f"{MODE_LABELS[mode]}  {value:.1f} %", xy=(1, y), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK,
            )
            ax.plot([1.0], [value], marker="o", ms=5, color=MODE_COLORS[mode],
                    transform=ax.get_yaxis_transform(), clip_on=False)


def plot_coverage(path: Path, curves: Curves, runs: list[str], maps: list[str]) -> None:
    n_steps = next(iter(curves.values())).shape[-1]
    episodes = next(iter(curves.values())).shape[1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        draw_coverage(ax, curves, runs, list(range(len(maps))), labels=True)
        coverage_axis(ax, n_steps)
        ax.set_xlabel(f"pas (échelle log, jusqu'à {n_steps - 1})")
        ax.set_ylabel("couverture (%)")
        header(fig, "Couverture au fil de l'épisode, toutes cartes", runs_text(runs, episodes))
        save(fig, path)


def plot_coverage_by_family(path: Path, curves: Curves, runs: list[str], maps: list[str]) -> None:
    groups = {g: idx for g, idx in map_groups(maps).items() if g != "all"}
    n_steps = next(iter(curves.values())).shape[-1]
    episodes = next(iter(curves.values())).shape[1]
    n_cols = min(4, len(groups))
    n_rows = -(-len(groups) // n_cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows + 0.8),
            sharex=True, sharey=True, squeeze=False,
        )
        for ax, (group, idx) in zip(axes.flat, groups.items()):
            draw_coverage(ax, curves, runs, idx, labels=False)
            coverage_axis(ax, n_steps)
            ax.set_title(f"{group}  ({len(idx)} carte{'s' * (len(idx) > 1)})", fontsize=10)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("couverture (%)")
        for ax in axes[-1]:
            ax.set_xlabel("pas (log)")
        header(fig, "Couverture au fil de l'épisode, par famille de cartes",
               runs_text(runs, episodes), modes_of(curves))
        save(fig, path)


def dot_panel(ax: plt.Axes, scores: dict[str, dict[str, np.ndarray]], order: list[str], title: str) -> None:
    modes = list(next(iter(scores.values())))
    offsets = {m: (k - (len(modes) - 1) / 2) * 0.22 for k, m in enumerate(modes)}
    y = {g: len(order) - 1 - i for i, g in enumerate(order)}
    for group in order:
        for mode in modes:
            values = scores[group][mode] * 100
            mean, _, low, high = mean_ci(values / 100)
            pos = y[group] + offsets[mode]
            color = MODE_COLORS[mode]
            if len(values) > 1:
                ax.scatter(values, np.full(len(values), pos), s=10, color=color, alpha=0.35, lw=0, zorder=2)
                ax.hlines(pos, low * 100, high * 100, color=color, lw=2, zorder=3)
            ax.scatter([mean * 100], [pos], s=46, color=color, edgecolor=SURFACE, lw=1.5, zorder=4)
            above = offsets[mode] >= 0
            ax.annotate(f"{mean * 100:.0f}", (mean * 100, pos), xytext=(0, 6 if above else -6),
                        textcoords="offset points", ha="center", va="bottom" if above else "top",
                        fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(list(y.values()), list(y))
    for label in ax.get_yticklabels():
        if label.get_text() == "all":
            label.set_fontweight("semibold")
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-3, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    if "all" in y:
        ax.axhline(y["all"] - 0.5, color="#c3c2b7", lw=0.8)


def plot_summary(path: Path, curves: Curves, runs: list[str], maps: list[str]) -> None:
    groups = map_groups(maps)
    episodes = next(iter(curves.values())).shape[1]
    coverage, solved = {}, {}
    for group, idx in groups.items():
        coverage[group], solved[group] = {}, {}
        for mode in modes_of(curves):
            final = np.stack([curves[run, mode] for run in runs])[..., -1]
            coverage[group][mode] = final.mean(2)[:, idx].mean(1)
            solved[group][mode] = (final >= 1).mean(2)[:, idx].mean(1)
    families = sorted((g for g in groups if g != "all"), key=lambda g: -coverage[g][modes_of(curves)[0]].mean())
    order = ["all", *families]
    with plt.rc_context(STYLE):
        fig, (ax_cov, ax_sol) = plt.subplots(
            1, 2, figsize=(11, 0.42 * len(order) + 1.8), sharey=True
        )
        dot_panel(ax_cov, coverage, order, "Couverture finale (%)")
        dot_panel(ax_sol, solved, order, "Épisodes explorés à 100 % (%)")
        text = runs_text(runs, episodes)
        if len(runs) > 1:
            text += " · petits points : un run"
        header(fig, "Résultats en fin d'épisode, par famille de cartes", text, modes_of(curves))
        save(fig, path)


def plot_steps_to_full(path: Path, rows: list[dict], runs: list[str]) -> None:
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    families = sorted({r["family"] for r in rows})
    offsets = {m: (k - (len(modes) - 1) / 2) * 0.3 for k, m in enumerate(modes)}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(families) + 1.8))
        for i, family in enumerate(families):
            y = len(families) - 1 - i
            for mode in modes:
                group = [r for r in rows if r["family"] == family and r["mode"] == mode]
                steps = np.array([r["steps_to_full"] for r in group if r["solved"]], dtype=float)
                pos = y + offsets[mode]
                color = MODE_COLORS[mode]
                label = f"{len(steps)}/{len(group)}"
                if len(steps):
                    q5, q25, q50, q75, q95 = np.percentile(steps, [5, 25, 50, 75, 95])
                    ax.hlines(pos, q5, q95, color=color, lw=1, alpha=0.7)
                    ax.hlines(pos, q25, q75, color=color, lw=6, alpha=0.85, capstyle="round")
                    ax.scatter([q50], [pos], s=26, color=SURFACE, edgecolor=color, lw=1.5, zorder=3)
                    ax.annotate(label, (q95, pos), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=7.5, color=INK_MUTED)
                else:
                    ax.annotate(f"aucun épisode à 100 % ({label})", (1, pos), xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=7.5, color=INK_MUTED)
        ax.set_xscale("log")
        ax.minorticks_off()
        ticks = [1, 10, 30, 100, 300, 1000, 2000]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(1, 2500)
        ax.set_yticks(range(len(families)), families[::-1])
        ax.set_ylim(-0.7, len(families) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("pas pour atteindre 100 % (log)")
        header(fig, "Vitesse d'exploration complète",
               "Épisodes résolus · barre : 25-75 %, trait : 5-95 %, rond : médiane · n résolus / total",
               modes)
        save(fig, path)


def plot_map_heatmap(path: Path, curves: Curves, runs: list[str], maps: list[str]) -> None:
    modes = modes_of(curves)
    order = sorted(range(len(maps)), key=lambda m: (map_family(maps[m]), maps[m]))
    values = np.array([
        [np.stack([curves[run, mode][m, :, -1] for run in runs]).mean() * 100 for mode in modes]
        for m in order
    ])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 0.36 * len(order) + 1.6))
        ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 60 else INK)
        ax.set_xticks(range(len(modes)), [MODE_LABELS[m] for m in modes])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(order)), [Path(maps[m]).stem for m in order], fontsize=8.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(1, len(order)):
            if map_family(maps[order[i]]) != map_family(maps[order[i - 1]]):
                ax.axhline(i - 0.5, color=SURFACE, lw=3)
        header(fig, "Couverture finale par carte (%)", f"Moyenne sur {len(runs)} run{'s' * (len(runs) > 1)}")
        save(fig, path)


def write_report(out: Path, curves: Curves, runs: list[str], maps: list[str]) -> None:
    episodes = episode_rows(curves, maps)
    summary, curve_rows = summary_and_curve_rows(curves, runs, maps)
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "coverage_per_map.csv", per_map_rows(curves, maps))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "coverage_curve.csv", curve_rows)
    write_markdown(out / "summary.md", summary)

    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    plot_coverage(plots / "coverage.png", curves, runs, maps)
    plot_coverage_by_family(plots / "coverage_by_family.png", curves, runs, maps)
    plot_summary(plots / "summary.png", curves, runs, maps)
    plot_steps_to_full(plots / "steps_to_full.png", episodes, runs)
    plot_map_heatmap(plots / "coverage_per_map.png", curves, runs, maps)
