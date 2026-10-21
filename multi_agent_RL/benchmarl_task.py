from __future__ import annotations

from collections.abc import Callable

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase

from multi_agent_RL.tunnel_env import GROUP, TunnelEnv
from multi_agent_RL.tunnel_map import TunnelMap


class TunnelTaskClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: int | None,
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = dict(self.config)
        maps = [TunnelMap.from_name(name) for name in config.pop("maps")]
        return lambda: TunnelEnv(
            maps, num_envs=num_envs, seed=seed, device=device, **config
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        n_agents = env.full_action_spec[GROUP, "action"].shape[-1]
        return {GROUP: [f"{GROUP}_{i}" for i in range(n_agents)]}

    def observation_spec(self, env: EnvBase) -> Composite:
        return self._group_spec(env, "observation")

    def info_spec(self, env: EnvBase) -> Composite | None:
        return self._group_spec(env, "info")

    def state_spec(self, env: EnvBase) -> Composite | None:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def action_mask_spec(self, env: EnvBase) -> Composite | None:
        return None

    def log_info(self, batch: TensorDictBase) -> dict[str, float]:
        done = batch.get(("next", "done")).squeeze(-1)
        if not done.any():
            return {}
        coverage = batch.get(("next", GROUP, "info", "coverage"))[done]
        return {"coverage_at_episode_end": coverage.float().mean().item()}

    @staticmethod
    def env_name() -> str:
        return "tunnel"

    def _group_spec(self, env: EnvBase, key: str) -> Composite:
        spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            for k in list(spec[group].keys()):
                if k != key:
                    del spec[group][k]
        return spec


class TunnelTask(Task):
    TUNNEL = None

    @staticmethod
    def associated_class():
        return TunnelTaskClass
