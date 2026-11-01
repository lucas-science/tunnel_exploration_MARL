from __future__ import annotations

import argparse
from pathlib import Path

import benchmarl.experiment.experiment as benchmarl_experiment
import torch
from benchmarl.algorithms import (
    IppoConfig,
    IqlConfig,
    MappoConfig,
    QmixConfig,
    VdnConfig,
)
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import GruConfig, MlpConfig

from multi_agent_RL.benchmarl_task import TunnelTask
from multi_agent_RL.tunnel_map import MAPS_DIR
from multi_agent_RL.util import run_name

ALGORITHMS = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
}
MODELS = {"mlp": MlpConfig, "gru": GruConfig}

RUNS_DIR = Path(__file__).parent / "runs"

benchmarl_experiment.generate_exp_name = run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    map_names = sorted(p.name for p in MAPS_DIR.glob("*.csv"))
    parser.add_argument("--maps", nargs="+", default=["map1.csv"], choices=map_names)
    parser.add_argument("--algo", default="mappo", choices=ALGORITHMS)
    parser.add_argument("--model", default="mlp", choices=MODELS)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--view-range", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument(
        "--reward-mode", default="grouped", choices=["grouped", "mixed"]
    )
    parser.add_argument("--reward-alpha", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--frames-per-batch", type=int, default=6000)
    parser.add_argument("--n-envs", type=int, default=120)
    parser.add_argument("--minibatch-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval-interval", type=int, default=120_000)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loggers", nargs="*", default=["csv", "tensorboard"])
    parser.add_argument("--sampling-device", default="cpu")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_experiment(args: argparse.Namespace) -> Experiment:
    task = TunnelTask.TUNNEL.get_task(
        config={
            "maps": args.maps,
            "n_agents": args.n_agents,
            "view_range": args.view_range,
            "max_steps": args.max_steps,
            "step_penalty": args.step_penalty,
            "reward_mode": args.reward_mode,
            "reward_alpha": args.reward_alpha,
        }
    )

    config = ExperimentConfig.get_from_yaml()
    config.train_device = args.device
    config.sampling_device = args.sampling_device
    config.buffer_device = args.sampling_device
    print(args.device)
    config.max_n_frames = args.frames
    config.on_policy_collected_frames_per_batch = args.frames_per_batch
    config.off_policy_collected_frames_per_batch = args.frames_per_batch
    config.on_policy_n_envs_per_worker = args.n_envs
    config.off_policy_n_envs_per_worker = args.n_envs
    config.on_policy_minibatch_size = min(args.minibatch_size, args.frames_per_batch)
    config.on_policy_n_minibatch_iters = args.epochs
    config.evaluation_interval = args.eval_interval
    config.render = False
    config.loggers = args.loggers
    config.checkpoint_interval = args.checkpoint_interval
    config.checkpoint_at_end = True
    RUNS_DIR.mkdir(exist_ok=True)
    config.save_folder = str(RUNS_DIR)

    return Experiment(
        task=task,
        algorithm_config=ALGORITHMS[args.algo].get_from_yaml(),
        model_config=MODELS[args.model].get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=config,
    )


def main() -> None:
    build_experiment(parse_args()).run()


if __name__ == "__main__":
    main()
