from __future__ import annotations

import argparse

import torch
from tensordict import TensorDict

from multi_agent_RL.tunnel_env import ACTIONS, GROUP, TunnelEnv
from multi_agent_RL.tunnel_map import TunnelMap

KEYS = {
    ".": "stay",
    "x": "stay",
    "z": "up",
    "w": "up",
    "s": "down",
    "q": "left",
    "a": "left",
    "d": "right",
}
HELP = (
    "zqsd / wasd: move, '.' or 'x': stay, one key per drone (e.g. 'zd.s') "
    "or one key for all, 'r': reset, 'o': toggle observations, Ctrl+D: quit"
)


def parse_actions(line: str, n_agents: int) -> list[int] | None:
    keys = line.replace(" ", "").lower()
    if len(keys) == 1:
        keys *= n_agents
    if len(keys) != n_agents or any(k not in KEYS for k in keys):
        return None
    return [ACTIONS.index(KEYS[k]) for k in keys]


def show(env: TunnelEnv, td: TensorDict, show_obs: bool) -> None:
    print(env.render())
    coverage = float(env.coverage()[0])
    print(f"step {int(env.n_steps[0])}/{env.max_steps}  coverage {coverage:.1%}")
    if show_obs:
        for i, obs in enumerate(td[GROUP, "observation"][0]):
            print(f"drone {i}\n{env.describe_observation(obs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="old_envs/map1.csv")
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--view-range", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--reward-mode", choices=["grouped", "mixed"], default="grouped"
    )
    parser.add_argument("--obs", action="store_true")
    args = parser.parse_args()

    env = TunnelEnv(
        TunnelMap.from_name(args.map),
        n_agents=args.n_agents,
        view_range=args.view_range,
        max_steps=args.max_steps,
        reward_mode=args.reward_mode,
    )
    td = env.reset()
    show_obs, total = args.obs, 0.0
    print(HELP)
    show(env, td, show_obs)

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line == "r":
            td, total = env.reset(), 0.0
        elif line == "o":
            show_obs = not show_obs
        elif (actions := parse_actions(line, env.n_agents)) is None:
            print(HELP)
            continue
        else:
            action_td = TensorDict(
                {GROUP: {"action": torch.tensor([actions])}}, batch_size=env.batch_size
            )
            td = env.step(action_td)["next"]
            rewards = td[GROUP, "reward"][0, :, 0].tolist()
            total += rewards[0]
            print("rewards", [round(r, 3) for r in rewards], f"total {total:.2f}")
        show(env, td, show_obs)
        if td["done"].item():
            reason = "fully explored" if td["terminated"].item() else "max steps"
            print(f"episode over ({reason}), 'r' to restart")


if __name__ == "__main__":
    main()
