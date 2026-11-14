from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from report import ERROR_KW, MODE_COLORS, MODE_LABELS, STYLE, save

RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "figures"
RL = {
    "ExperienceClassiqueMLP": "MLP",
    "ExperienceMLPHotVector": "MLP + ID",
    "ExperienceClassiqueMLPMixte": "MLP, mixed",
    "ExperienceMLPHotVectorMixte": "MLP + ID, mixed",
}
BASELINES = {
    "Baseline_random": "Random",
    "Baseline_random_forward": "Random forward",
    "Baseline_left_hand": "Left wall",
    "Baseline_right_hand": "Right wall",
}
MAP_SETS = {"small_maps": "small", "big_maps": "big"}
RL_MODES = ["greedy", "stochastic"]


def load() -> dict[tuple[str, str, str], tuple[float, float]]:
    scores = {}
    for exp in [*RL, *BASELINES]:
        with open(RUNS / exp / "evaluation" / "summary.csv") as f:
            for r in csv.DictReader(f):
                if r["group"] == "all":
                    std = float(r["coverage_std"]) if r["coverage_std"] not in ("", "nan") else np.nan
                    scores[exp, r["maps"], r["mode"]] = (100 * float(r["coverage_mean"]), 100 * std)
    return scores


def bar(ax: plt.Axes, x: float, value: tuple[float, float], width: float, **kw) -> None:
    mean, std = value
    ax.bar(x, mean, width, yerr=None if np.isnan(std) else std, error_kw=ERROR_KW, **kw)


def mode_legend(fig: plt.Figure, extra: list[Patch] = ()) -> None:
    handles = [Patch(color=MODE_COLORS[m], label=MODE_LABELS[m]) for m in [*RL_MODES, "baseline"]]
    fig.legend(handles=[*handles, *extra], loc="upper center", ncol=len(handles) + len(extra),
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2)


def plot_final_coverage(scores: dict) -> None:
    width = 0.38
    labels = [*RL.values(), *BASELINES.values()]
    x = np.arange(len(labels))
    for maps, suffix in MAP_SETS.items():
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(3.6, 2.8))
            for i, exp in enumerate(RL):
                for k, mode in enumerate(RL_MODES):
                    bar(ax, i + (k - 0.5) * width, scores[exp, maps, mode], width, color=MODE_COLORS[mode])
            for i, exp in enumerate(BASELINES, start=len(RL)):
                bar(ax, i, scores[exp, maps, "baseline"], 2 * width, color=MODE_COLORS["baseline"])
            ax.axvline(len(RL) - 0.5, color="0.6", lw=0.8, ls=":")
            ax.set_xticks(x, labels, rotation=45, ha="right")
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.set_ylim(0, 105)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("Final coverage (%)")
            mode_legend(fig)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            save(fig, OUT / f"final_coverage_{suffix}")


def plot_ablation(scores: dict, maps: str = "big_maps") -> None:
    rewards = {"Grouped": ("ExperienceClassiqueMLP", "ExperienceMLPHotVector"),
               "Mixed": ("ExperienceClassiqueMLPMixte", "ExperienceMLPHotVectorMixte")}
    width = 0.2
    random_mean = scores["Baseline_random", maps, "baseline"][0]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for i, (no_id, with_id) in enumerate(rewards.values()):
            for k, mode in enumerate(RL_MODES):
                for j, exp in enumerate((no_id, with_id)):
                    pos = i + (2 * k + j - 1.5) * width
                    bar(ax, pos, scores[exp, maps, mode], width * 0.92, color=MODE_COLORS[mode],
                        alpha=1.0 if j else 0.45, lw=0)
        ax.axhline(random_mean, color="0.35", lw=1, ls="--")
        ax.set_xticks(range(len(rewards)), [f"{r} reward" for r in rewards])
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Final coverage (%)")
        handles = [Patch(color=MODE_COLORS[m], alpha=a, lw=0, label=f"{MODE_LABELS[m]}{s}")
                   for m in RL_MODES for a, s in ((0.45, ""), (1.0, " + ID"))]
        handles.append(plt.Line2D([], [], color="0.35", lw=1, ls="--", label="Random"))
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        save(fig, OUT / "ablation_id_reward")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = load()
    plot_final_coverage(data)
    plot_ablation(data)
