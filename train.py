from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import benchmarl.experiment.experiment as benchmarl_experiment
import torch
from benchmarl.algorithms import (
    IppoConfig,
    IqlConfig,
    MappoConfig,
    QmixConfig,
    VdnConfig,
)
from benchmarl.experiment import ExperimentConfig
from benchmarl.models import CnnConfig, GruConfig, MlpConfig

from benchmarl_task import TunnelExperiment, TunnelTask
from tunnel_map import resolve_maps
from util import run_name

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
    parser.add_argument("--maps", nargs="+", default=["small_maps/train"])
    parser.add_argument("--eval-maps", nargs="*", default=["small_maps/val"])
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
    parser.add_argument("--hot-vector", action="store_true")
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
    args = parser.parse_args()
    if args.frames_per_batch % args.n_envs:
        parser.error(
            f"--frames-per-batch ({args.frames_per_batch}) must be a multiple of "
            f"--n-envs ({args.n_envs}), e.g. {args.n_envs * 50}"
        )
    for name in ("eval_interval", "checkpoint_interval"):
        if getattr(args, name) % args.frames_per_batch:
            parser.error(
                f"--{name.replace('_', '-')} ({getattr(args, name)}) must be a "
                f"multiple of --frames-per-batch ({args.frames_per_batch})"
            )
    return args


def build_experiment(args: argparse.Namespace) -> TunnelExperiment:
    eval_maps = resolve_maps(args.eval_maps)
    task = TunnelTask.TUNNEL.get_task(
        config={
            "maps": resolve_maps(args.maps),
            "eval_maps": eval_maps,
            "n_agents": args.n_agents,
            "view_range": args.view_range,
            "max_steps": args.max_steps,
            "step_penalty": args.step_penalty,
            "reward_mode": args.reward_mode,
            "reward_alpha": args.reward_alpha,
            "hot_vector": args.hot_vector,
            "global_state": args.algo == "mappo",
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
    if eval_maps:
        config.evaluation_episodes = len(eval_maps)
    config.render = False
    config.loggers = args.loggers
    config.checkpoint_interval = args.checkpoint_interval
    config.checkpoint_at_end = True
    config.exclude_buffer_from_checkpoint = True
    RUNS_DIR.mkdir(exist_ok=True)
    config.save_folder = str(RUNS_DIR)

    algorithm_config = ALGORITHMS[args.algo].get_from_yaml()
    if hasattr(algorithm_config, "minibatch_advantage"):
        algorithm_config.minibatch_advantage = True

    critic_model_config = MlpConfig.get_from_yaml()
    if args.algo == "mappo":
        critic_model_config = CnnConfig.get_from_yaml()
        critic_model_config.cnn_strides = 2
        critic_model_config.mlp_num_cells = [256]

    return TunnelExperiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=MODELS[args.model].get_from_yaml(),
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=config,
    )


def main() -> None:
    build_experiment(parse_args()).run()


if __name__ == "__main__":
    main()
