# Multi-agent RL for tunnel exploration

A fleet of drones learns to explore tunnel maps cooperatively with
[BenchMARL](https://github.com/facebookresearch/BenchMARL) (MAPPO, QMIX, ...).
The environment is a vectorised TorchRL env: it simulates many episodes at once.

## Train

```bash
python -m multi_agent_RL.train --maps map1.csv map2.csv
```

| Option | Default | |
| --- | --- | --- |
| `--algo` | `mappo` | `mappo`, `ippo`, `qmix`, `vdn`, `iql` |
| `--model` | `mlp` | `mlp`, or `gru` (memory, slower) |
| `--maps` | `map1.csv` | one or more maps from `envs/`, they will be uniformly picked |
| `--n-agents` / `--view-range` / `--max-steps` | 4 / 3 / 500 | environment |
| `--reward-mode` / `--reward-alpha` | `grouped` / 0.5 | reward |
| `--frames` | 3 000 000 | training budget |
| `--n-envs` | 120 | episodes simulated in parallel |
| `--checkpoint-interval` | 0 | save every N frames (multiple of `--frames-per-batch`) |
| `--loggers` | `csv tensorboard` | also `wandb` (needs `pip install wandb` + `wandb login`) |

Results go to `runs/<run>/`: one CSV per metric (the key one is
`coverage_at_episode_end`) and `checkpoints/checkpoint_<frames>.pt`.
Without `--checkpoint-interval`, the checkpoint is only written at the end of the run.

## Training curves

```bash
pip install tensorboard   # once
tensorboard --logdir multi_agent_RL/runs # Open http://localhost:6006. 
```

## Visualise a trained policy

```bash
python -m multi_agent_RL.visualize multi_agent_RL/runs/<run> --map env_cross.csv
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

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest multi_agent_RL
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is only needed if ROS is installed on the machine.

## Files

| File | Content |
| --- | --- |
| `tunnel_map.py` | Load a CSV map |
| `tunnel_env.py` | Vectorised TorchRL environment |
| `benchmarl_task.py` | Plugs the environment into BenchMARL |
| `train.py` | Training script |
| `visualize.py` | Replay a checkpoint as a GIF |
