# Multi-agent RL for tunnel exploration

A fleet of drones learns to explore tunnel maps cooperatively with
[BenchMARL](https://github.com/facebookresearch/BenchMARL) (MAPPO, QMIX, ...).
The environment is a vectorised TorchRL env: it simulates many episodes at once.

## Maps

Two map sets, `envs/small_maps/` (every map 15×15, used for training by default) and `envs/big_maps/` (every map 50×50), with the same families and the same splits:

| Folder | Maps (small / big) | Usage |
| --- | --- | --- |
| `<set>/train/` | 70 / 72 | training |
| `<set>/val/` | 14 / 14 (2 per train family) | evaluation during training, hyperparameter and reward tuning |
| `<set>/test/` | 15 / 15 | final evaluation only, never used to tune anything. The 7 train families plus 8 unseen ones (spiral, binary tree, tiny dead end, ...), one map each, with the same file names in both sets |
| `envs/old_envs/` | 6 | first hand-made maps |

`<set>/catalog.csv` describes every map and its split.

## What each network sees

Each drone has a heading: its actions are `stay`, `forward`, `left`, `right`, `back`, relative to it. The heading becomes the direction of the drone's last actual move, and is drawn at random for each drone at the start of an episode.

Training is centralised, execution is decentralised:

| Network | Input | Used at inference |
| --- | --- | --- |
| Actor | 4 rays of `--view-range` cells (`blocked`, `drone`) pointing forward, left, right and back relative to the drone's heading, shared-cell flag, optional drone ID. It never sees which cells were explored | yes |
| MAPPO critic | global map as an image (`walls`, `explored`, `drones`), through a CNN | no |

Other algorithms use the actor observations only.

## Train

```bash
python -m train
# full GPU, about 2× faster
python -m train \
    --sampling-device cuda --n-envs 512 --frames-per-batch 25600 --eval-interval 204800
# with the drone ID and a checkpoint every 8 iterations
python -m train --sampling-device cuda \
    --n-envs 512 --frames-per-batch 25600 --eval-interval 204800 \
    --checkpoint-interval 204800 --hot-vector
```

`--frames-per-batch` must be a multiple of `--n-envs`, and `--eval-interval` /
`--checkpoint-interval` multiples of `--frames-per-batch`. The script refuses to
start otherwise: a mismatch would make the collector round the batch size and the
run crash at the end, before the final checkpoint.

| Option | Default | |
| --- | --- | --- |
| `--algo` | `mappo` | `mappo`, `ippo`, `qmix`, `vdn`, `iql` |
| `--model` | `mlp` | `mlp`, or `gru` (memory, slower) |
| `--maps` | `train` | maps or folders, relative to `envs/`, uniformly picked during training |
| `--eval-maps` | `val` | maps or folders used by the evaluation; each map is played once per evaluation. Empty (`--eval-maps`) to evaluate on the training maps |
| `--n-agents` / `--view-range` / `--max-steps` | 4 / 3 / 500 | environment. An episode is also truncated when the team discovers no new cell for as many steps as the map's diameter (the longest shortest path between two reachable cells), so a drone always has time to cross the map |
| `--reward-mode` / `--reward-alpha` | `grouped` / 0.5 | reward |
| `--hot-vector` | not set | flag, no value: add `--hot-vector` to enable. Appends the drone's one-hot ID to its observation, so drones can learn different roles (fixes the number of agents) |
| `--frames` | 3 000 000 | training budget |
| `--n-envs` | 120 | episodes simulated in parallel. On CPU, ~120 is enough (more barely speeds up). On GPU (`--sampling-device cuda`), the MAPPO global map limits the batch size, which is held twice on the training device: on a 4 GB GPU, use 512 envs × 50 steps (about 3 GB reserved); 768 fits but is tight, 1024 runs out of memory. The replay buffer stays in RAM; run only one training at a time on the GPU |
| `--frames-per-batch` | 6000 | frames collected per iteration; keep it equal to `--n-envs` × 50 (at least 25 steps per env). `--eval-interval` and `--checkpoint-interval` must be multiples of it |
| `--sampling-device` | `cpu` | `cuda` runs the env and the collection on GPU; worth it from ~1024 envs |
| `--eval-interval` | 120 000 | evaluate every N frames (deterministic policy, logged as `eval/coverage_at_episode_end`); must be a multiple of `--frames-per-batch` |
| `--checkpoint-interval` | 0 | save every N frames (multiple of `--frames-per-batch`) |
| `--seed` | 0 | one or more seeds; one run per seed, named `..._seed<N>` |
| `--experiment` | empty | subfolder of `runs/` to save the run in, e.g. `Experience1` |
| `--loggers` | `csv tensorboard` | also `wandb` (needs `pip install wandb` + `wandb login`) |

Results go to `runs/<run>/`: one CSV per metric (the key one is
`coverage_at_episode_end`) and `checkpoints/checkpoint_<frames>.pt`.
Without `--checkpoint-interval`, the checkpoint is only written at the end of the run,
so an interrupted or crashed run loses its policy.

## Training curves

```bash
pip install tensorboard   # once
tensorboard --logdir runs # Open http://localhost:6006. 
```

## Evaluate an experiment

Put the runs of one experiment in the same folder, one training seed per run:

```bash
python -m train --experiment Experience1 --seed 0 1 2 3 4   # one run per seed, one after the other
python -m evaluate runs/Experience1 --maps small_maps/val big_maps/val
```

Each run is played in two modes, `greedy` (most likely action) and `stochastic`
(sampled action). An episode ends when every reachable cell is explored, or when no new cell is found for
`--patience` × the map diameter steps (`--max-steps` is only a safety cap). Coverage is
measured on the cells reachable from the start. Results go to
`runs/Experience1/evaluation/` (or `runs/<run>/evaluation/` when given a single run):

Results are split by map set (`small_maps`, `big_maps`: the first folder of the map path), since a policy can behave very differently on larger maps.

| File | Content |
| --- | --- |
| `summary.md` / `summary.csv` | an overview with the three metrics of each map set side by side, then for each set, per mode, over all maps then per map family: final coverage, share of episodes reaching 100 % and 90 %, and steps to reach 90 % coverage (exploration speed, averaged over the episodes that reach 90 %) |
| `coverage_curve.csv` | mean coverage per step per mode and group, with its 95 % CI |
| `coverage_per_map.csv` | mean coverage per step for each run, mode and map |
| `episodes.csv` | length, final coverage, solved flag, steps to 100 % and steps to 90 % (empty if never reached) of every episode |
| `plots/<set>/coverage.png` | mean coverage per step over all maps, mean ± std band over runs, log step axis |
| `plots/<set>/coverage_by_family.png` | the same curve, one panel per map family |
| `plots/<set>/final_coverage.png` | final coverage per map family and mode, bars with ± std over runs |
| `<run>/evaluation/animation/<set>/<family>/<map>_<mode>.gif` | one per family and mode of the `--gif-maps` sets (first episode of the family's first map): map, drone trails and coverage curve, cut 10 steps after the last discovery |

Each run is first averaged per map, then over the maps of a group. Tables show the mean ± standard deviation over runs (`summary.csv` also keeps the Student 95 % CI, used by the plots).
`python -m report runs/<experiment>/evaluation` rebuilds `summary.md` / `summary.csv` from `episodes.csv` without re-running the episodes. With a single run (an experiment folder holding one run, or a run folder), the std and CI are left empty.

| Option | Default | |
| --- | --- | --- |
| `--maps` | `small_maps/val big_maps/val` | maps or folders, relative to `envs/`. Use `test` maps only for the final evaluation |
| `--episodes` | 30 | episodes per map and per run |
| `--patience` | 2 | an episode stops after `patience` × map diameter steps without discovering a cell |
| `--max-steps` | 10000 | safety cap on the episode length |
| `--seed` | 0 | the same seed for every run, so all runs start from the same initial headings |
| `--fps` | 5 | GIF speed |
| `--no-gif` | off | skip the GIFs. Otherwise they are written for the first run only (seed 0) |
| `--gif-maps` | `small_maps` | map sets that get GIFs (one per family and mode) |
| `--workers` | 8 | parallel processes. Each (run, mode) pair is split into chunks of maps so that a single run or a baseline also uses every worker; the GIFs are rendered in the same pool |

## Evaluate a baseline algorithm

`evaluate.py` also takes the name of a non-learned algorithm from `algo/` instead of a folder:

```bash
python -m evaluate tremaux --n-agents 4   # writes runs/Baseline_tremaux/evaluation/
```

The algorithm plays the same maps, episodes, seed, collisions and coverage measure as the policies, in a single mode `baseline`, and produces the same files (GIFs in `evaluation/animation/`).
A drone whose move was blocked by another drone takes a random legal move with probability 0.5, so that two drones facing each other in a corridor do not stay stuck.

| Algorithm | Behaviour |
| --- | --- |
| `random` | uniform legal move, `stay` only when no move is possible (Q1 reference) |
| `random_forward` | goes forward whenever it can, otherwise left or right at random, and back only in a dead end |
| `left_hand` / `right_hand` | goes straight until it touches a wall, then keeps it on its left (right) |
| `tremaux` | Trémaux: marks each passage it takes, turns back when it reaches an already visited cell through a new passage, otherwise takes the least marked passage. Each drone keeps its own marks |
| `tremaux_shared` | the same with marks shared by the whole fleet |
| `frontier` | goes to the nearest known cell it has not visited, along known cells; each drone only knows what it has seen with its 4 rays (Q2 individual map baseline) |
| `frontier_shared` | the same with a map shared by the whole fleet (Q2 shared map baseline) |

| Option | Default | |
| --- | --- | --- |
| `--n-agents` / `--view-range` | 4 / 3 | environment of the algorithm |
| `--experiment` | `Baseline_<algo>` | output folder in `runs/` |

To add an algorithm, subclass `algo.base.Algorithm`: `reset()` initialises its memory after the environment reset, `decide(td)` returns one action per drone (`(num_envs, n_agents)`, same egocentric actions as the policies) and can read the environment through `self.env`. Then register it in `ALGORITHMS` in `algo/__init__.py`.

## Play the environment by hand

```bash
python -m play --map old_envs/env_cross.csv --n-agents 2 --obs
```

Type one key per drone, relative to its heading (`z`/`w` forward, `q`/`a` left, `d` right, `s` back, `.` to stay), e.g. `d.` turns drone 0
to its right and keeps drone 1 still. Each drone's heading is printed under the map. Drones move at the same time: a drone can only
enter a free cell or one left during the same step. When several drones aim at
the same cell, a random one gets it; two drones aiming at each other both stay. Drones are shown by their index (`+` on the
start cell while several drones are still there). `--obs` prints what each drone sees along its 4 relative rays:
`#` blocked, `D` drone, `.` free.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is only needed if ROS is installed on the machine.

## Files

| File | Content |
| --- | --- |
| `tunnel_map.py` | Load a CSV map. First line: start column, start row, map diameter; then one row per line (`1` wall, `0` tunnel) |
| `tunnel_env.py` | Vectorised TorchRL environment |
| `benchmarl_task.py` | Plugs the environment into BenchMARL |
| `train.py` | Training script |
| `evaluate.py` | Evaluate the runs of an experiment or a baseline algorithm, with GIFs |
| `algo/` | Non-learned baseline algorithms (random, wall following, Trémaux, frontiers) |
| `report.py` | CSV, summary and plots of an evaluation |
