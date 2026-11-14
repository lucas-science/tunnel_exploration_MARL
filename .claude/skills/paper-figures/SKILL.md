---
name: paper-figures
description: House style for every figure of this project (evaluation plots, comparison figures, report figures). Use before writing or changing any matplotlib code here, e.g. a new plot in report.py, a figure comparing experiments or baselines, or a figure for doc/resultats.
---

# Figures in research-paper style

The user picked the "modern RL paper" style (seaborn-like, as in recent deep RL papers). The code of reference is `report.py`: reuse its constants and helpers instead of redefining them.

```python
from report import STYLE, MODE_COLORS, MODE_LABELS, FAMILY_LABELS, family_label, top_legend, save

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ...
    top_legend(fig, modes)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, path)  # writes path.png
```

## Rules

| Element | Rule |
| --- | --- |
| Title | none inside the image: the LaTeX caption explains the figure. Subplot titles only to name a panel (map family) |
| Panel letters | bold `a`, `b`, `c` at the top left when a figure has several independent panels (`ax.text(-0.16, 1.04, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")`) |
| Legend | one shared legend, centered above the panels, no frame (`top_legend`). No values written next to curves or points |
| Font | sans-serif (Nimbus Sans, close to Helvetica), 9 pt labels, 8 pt ticks and legend |
| Axes | left and bottom spines only, light grey grid (`0.9`), white background |
| Size | sized for a paper: 3.4 in wide for one column, 6.9 in for two columns, about 2.5 in high per row of panels |
| Colors | `MODE_COLORS`: greedy `#4C72B0`, stochastic `#DD8452`, baseline `#55A868`. A new series takes the next seaborn "deep" color (`#C44E52`, `#8172B3`, `#937860`) |
| Dispersion | mean ± standard deviation over runs (training seeds), as a shaded band (alpha 0.2) for curves, and for bars error bars with marked caps (`error_kw=report.ERROR_KW`: caps of 3 pt, 1 pt lines, dark grey). No 95 % CI on figures. With a single run, no band |
| Step axis | log scale from 1 to `max_steps`, label "Environment steps" (or "Steps" on small panels) |
| Language | English labels; short family names through `family_label` (`corner`, `braided`, `grid`) |
| Format | PNG only, 300 dpi, `bbox_inches="tight"` (`save`). Never PDF |

## What the user rejected

Do not bring these back:
- big title and subtitle inside the image, cream background, values annotated on curves or dots;
- the dot plot summary with 95 % CI (`summary.png`), the boxplot of steps to 100 % (`steps_to_full.png`), the per-map heatmap (`coverage_per_map.png`), the per-map-set overview (`overview.png`), per-map curve grids, the per-episode final-coverage boxplot;
- Plotly / HTML plots;
- PDF output.

Tables (`summary.md`) show `mean ± std` over runs, never intervals.

## Current evaluation figures

`report.write_plots` writes, per map set in `evaluation/plots/<set>/`: `coverage.png`, `coverage_by_family.png`, `final_coverage.png`. `python -m report runs/<experiment>/evaluation` rebuilds them from the CSVs in seconds, so a style change never needs a new evaluation.
