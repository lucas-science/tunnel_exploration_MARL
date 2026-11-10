# Multi-agent RL for tunnel exploration

A fleet of drones learns to explore tunnel maps cooperatively with
[BenchMARL](https://github.com/facebookresearch/BenchMARL) (MAPPO, QMIX, ...).
The environment is a vectorised TorchRL env: it simulates many episodes at once.

## Maps

| Folder | Maps | Usage |
| --- | --- | --- |
| `envs/train/` | 73 | training |
| `envs/val/` | 14 (2 per train family) | evaluation during training, hyperparameter and reward tuning |
| `envs/test/` | 16 | final evaluation only, never used to tune anything. Mixes train families and unseen ones (spiral, binary tree, 100×100, ...) |
| `envs/old_envs/` | 6 | first hand-made maps |

`envs/catalog.csv` describes every map and its set.

## What each network sees

Training is centralised, execution is decentralised:

| Network | Input | Used at inference |
| --- | --- | --- |
| Actor | 4 rays of `--view-range` cells (`blocked`, `drone`), shared-cell flag, optional drone ID. It never sees which cells were explored | yes |
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

## Visualise a trained policy

```bash
python -m visualize runs/<run> --map old_envs/env_cross.csv
```

Writes `runs/<run>/policy_<map>.gif` (map + coverage curve).

| Option | Default | |
| --- | --- | --- |
| `--map` | random among the run's maps | map to replay, can be unseen during training |
| `--stochastic` | off | sample actions from the policy probabilities, as during training, instead of taking the most likely one |
| `--seed` | random (printed) | replay the same episode |
| `--live` | off | open a window instead of writing a GIF |
| `--output` | `runs/<run>/policy_<map>.gif` | GIF path |
| `--fps` | 10 | GIF speed |

All drones start on the same cell with the same observation, so without
`--stochastic` they move together as one. Comparing both modes shows whether the
policy really spreads the drones or relies on randomness to do it.

## Play the environment by hand

```bash
python -m play --map old_envs/env_cross.csv --n-agents 2 --obs
```

Type one key per drone (`zqsd` or `wasd`, `.` to stay), e.g. `d.` moves drone 0
right and keeps drone 1 still. Drones move at the same time: a drone can only
enter a free cell or one left during the same step. When several drones aim at
the same cell, a random one gets it; two drones aiming at each other both stay. Drones are shown by their index (`+` on the
start cell while several drones are still there). `--obs` prints what each drone sees along its 4 rays:
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
| `visualize.py` | Replay a checkpoint as a GIF |
