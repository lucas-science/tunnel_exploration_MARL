---
name: beamer-slides
description: Style and workflow for the LaTeX Beamer slides of this project (points d'étape, presentations of results). Use whenever the user asks to create, edit or extend slides, a presentation or a "point d'étape" in doc/slides/, e.g. "fais les slides du point 4", "ajoute une slide sur les résultats".
---

# Slides LaTeX simples et efficaces

The reference deck is `doc/slides/point_etape_3/main.tex`. A new deck starts from [template.tex](template.tex) (same preamble), in `doc/slides/<name>/main.tex` with its images in `doc/slides/<name>/figures/`.

## Preamble

- `beamer`, `aspectratio=169`, `11pt`, theme `metropolis` with `progressbar=frametitle,numbering=fraction`
- Packages: `fontenc` T1, `lmodern`, `booktabs`, `graphicx`, `tikz` (`arrows.meta`). Add `animate` only for an animated GIF
- `\graphicspath{{figures/}}`, numbered sections in the table of contents, custom section page from the template
- Title page, then an "Au programme" frame with `\tableofcontents[hideallsubsections]`

## Content rules

| Element | Rule |
| --- | --- |
| Language | French, with accents written directly (UTF-8) |
| Frame title | states the message of the slide ("Plus de superposition entre drones"), not just a topic ("Environnement") |
| Text | 3 to 5 bullets per frame, one idea each, short sentences without final period. No paragraph |
| Density | one message per frame. If it does not fit at `\small`, split the frame rather than going below `\footnotesize` |
| Layout | `columns[c]` with ~0.52 / 0.46 `\textwidth` for figure + text; full-page figure with `height=0.84\textheight` |
| Figures | PNG from the project (`report.py`, `paper-figures` skill) copied into `figures/`. No title in the image |
| Diagrams | small TikZ drawings (grids, networks, flows) with soft colours (`blue!20`, `orange!25`, `green!30`), `font=\footnotesize`; repeated shapes go in a `\newcommand` in the preamble |
| Tables | `booktabs` only (`\toprule`, `\midrule`, `\bottomrule`), no vertical lines |
| References | numbered `[1]` in the text, full reference at the bottom of the frame in `\scriptsize` after `\vfill` |
| Notes | secondary remark at the bottom of the frame in `\footnotesize` after `\vfill` |
| Numbers | only values measured in `runs/` or `doc/`, never invented; percentages as `30 \%` |
| End | a "Suite du travail" frame with the next steps (`enumerate`, open questions allowed) |

Avoid: overlays and `\pause`, blocks (`block`, `alertblock`), colourful boxes, logos, more than one figure + one list per frame, maths when a sentence is enough.

## Build

```bash
cd doc/slides/<name> && latexmk -pdf -interaction=nonstopmode main.tex
```

After building, check the log for `Overfull` warnings (`grep -n Overfull main.log`) and fix every frame that overflows. The `Overfull \vbox (15.6pt too high)` at `\maketitle` comes from metropolis's title page and can be ignored. Copy the deck's `.gitignore` from `point_etape_3` so aux files stay out of git. If the user writes the text, the `style-francais` skill sets the tone.
