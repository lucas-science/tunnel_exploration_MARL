# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Multi-agent RL research project: a drone fleet learns to explore tunnel maps cooperatively, using BenchMARL on a vectorised TorchRL environment. `README.md` has the full list of CLI options.

## Commands

Dependencies live in `.venv/` (no requirements file or lint config in the repo). Scripts are run as modules from the repo root.

```bash
# tests (the env var avoids a crash from the ROS pytest plugins installed on this machine)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_tunnel_env.py::test_load_map

.venv/bin/python -m train                                  # CPU sampling defaults
.venv/bin/python -m train --sampling-device cuda --n-envs 512 --frames-per-batch 25600 --eval-interval 204800
.venv/bin/python -m evaluate runs/<experiment>   # small_maps/val + big_maps/val, writes runs/<experiment>/evaluation/
.venv/bin/python -m evaluate tremaux --n-agents 4                     # baseline from algo/, writes runs/Baseline_tremaux/evaluation/
.venv/bin/python -m play --map val/<map>.csv --n-agents 2 --obs
```

`train.py` refuses to start unless `--frames-per-batch` is a multiple of `--n-envs`, and `--eval-interval` / `--checkpoint-interval` are multiples of `--frames-per-batch`. Without `--checkpoint-interval`, the only checkpoint is written at the end of the run.

`envs/old_envs/` is referenced by the README, by `play.py`'s default `--map` and by 3 tests (`test_specs`, `test_envs_are_independent`, `test_rollout_auto_resets`), but is missing from the repo, so those tests fail with `FileNotFoundError`.

## Architecture

The flow is `tunnel_map.py` (CSV to `TunnelMap`) → `tunnel_env.py` (`TunnelEnv`, batched simulation) → `benchmarl_task.py` (BenchMARL adapter) → `train.py` / `evaluate.py` (runs the policies) → `report.py` (CSV, `summary.md`, PNG plots). `play.py` drives `TunnelEnv` directly.

### Map format and coordinates

- The first CSV line is `start_col,start_row,diameter`, while `TunnelMap.start` is `(row, col)`.
- `diameter` is read from the file, not computed. It sets the stall truncation limit, so it must be kept correct when a map is edited.
- `set_maps` pads all maps with walls up to the largest map plus `view_range` on each side, which keeps rays and moves in bounds. `TunnelEnv.positions` is in padded coordinates; use `agent_positions()` or `snapshot()` for map coordinates.

### Environment

All state is batched tensors of shape `(num_envs, ...)`, with per-agent data under the `"drone"` group. Keep new logic vectorised.

- **Actions and observations are egocentric**: each drone stores a `headings` entry (up/right/down/left), set to the direction of its last actual move and drawn at random on reset. Actions are `stay, forward, left, right, back` and the rays point forward, left, right, back relative to the heading. Tests fix headings with `face()` before checking moves or rays.
- **Actor observations are local and map-size independent**: 4 relative rays of `view_range` cells (`blocked`, `drone`), a shared-cell flag, and an optional one-hot ID. The actor deliberately never sees which cells were explored. This is why a policy can run on unseen maps of any size.
- **Global state is MAPPO-only**: `train.py` sets `global_state` when `--algo mappo`, and the CNN critic gets a `(H, W, 3)` boolean image of walls, explored and drones. `state_shape` is fixed when specs are built, and after `set_maps` with other dimensions `_global_map` returns zeros. That is harmless at evaluation because the critic is not used.
- **Move resolution** (`_move_agents`) is simultaneous: head-on swaps are blocked, contested cells go to a random priority winner, and a loop over `n_agents` propagates blocked chains.
- **Episode end**: `terminated` means fully explored. `truncated` means `max_steps` reached, or no new cell found for `diameter` steps.
- **Action masks** only rule out moves into walls.

### BenchMARL integration

- `TunnelExperiment` subclasses `Experiment`. `_setup_task` swaps the test env to the eval maps with `fixed_maps=True`, so each env index plays a fixed map and `evaluation_episodes = len(eval_maps)` covers each map once. `_setup_algorithm` forces on-policy buffer storage to CPU to save GPU memory.
- `train.py` starts from BenchMARL's YAML configs and overrides them in `build_experiment`. It also monkeypatches `generate_exp_name` to get the `runs/<timestamp>_...` folder names.
- `evaluate.py` does not go through BenchMARL's rollout: it builds its own `TunnelEnv` from the run's task config, disables the env's stall truncation, ends each episode itself when every reachable cell is explored or after `--patience` × diameter steps without discovery (curves are frozen after the end), and normalises coverage by the cells reachable from the start (protocol of `doc/protocole.md`).
- `evaluate.py` runs every (run, mode, chunk of maps) job in a `spawn` process pool (`--workers`) and renders the GIFs in the same pool; workers report steps through a shared counter. It has two evaluators sharing `run_episodes`: `PolicyEvaluator` (checkpoint, modes greedy and stochastic) and `AlgorithmEvaluator` (an `algo/` baseline, single mode `baseline`). Baselines subclass `algo.base.Algorithm` and implement `decide(td)` with batched tensors; `act` adds a random escape when a drone was blocked by another one, since head-on swaps are never resolved by the environment.
- `evaluate.py` aliases `multi_agent_RL.benchmarl_task` in `sys.modules` so checkpoints pickled before the project moved to the repo root still load. Keep this alias, and note that renaming or moving `benchmarl_task.py` breaks checkpoint reloading.

## Git

- Never mention Claude or Claude Code in commits or PRs: no `Co-Authored-By` trailer, no "Generated with Claude Code" line.

## Research conventions

- Maps live in `envs/small_maps/` (all 15x15, training defaults) and `envs/big_maps/` (all 50x50), each split into `train/`, `val/` and `test/`, with the same families in both sets (test files have the same names). Keep every map of a set at the set's size. `train/` is for training, `val/` for tuning and evaluation during training. `test/` is for final evaluation only and must never be used to tune anything.
- The key metric is `coverage_at_episode_end` (`eval/coverage_at_episode_end` on the validation maps).
- `runs/` is gitignored. Runs of one experiment go in `runs/<experiment>/` (`train.py --experiment`), with the evaluation in `runs/<experiment>/evaluation/`.
