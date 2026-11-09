from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs.utils import check_env_specs

from multi_agent_RL import TunnelEnv, TunnelMap
from multi_agent_RL.tunnel_env import GROUP
from multi_agent_RL.tunnel_map import map_names

SMALL_MAP_CSV = """\
2,1,4,,
1,1,1,1,1
1,0,0,0,1
1,0,0,0,1
1,0,0,0,1
1,1,1,1,1
"""


@pytest.fixture
def small_map(tmp_path):
    path = tmp_path / "small.csv"
    path.write_text(SMALL_MAP_CSV, encoding="utf-8")
    return TunnelMap.from_csv(path)


def act(env, actions):
    td = TensorDict(
        {GROUP: {"action": torch.tensor(actions)}}, batch_size=env.batch_size
    )
    return cast(TensorDictBase, env.step(td).get("next"))


def rewards_of(out):
    return cast(torch.Tensor, out[GROUP, "reward"]).flatten().tolist()


def rays_of(obs, view_range):
    return obs[:-1].reshape(4, view_range, 3)


def test_load_map(small_map):
    assert small_map.shape == (5, 5)
    assert small_map.start == (1, 2)
    assert small_map.n_tunnels == 9


@pytest.mark.parametrize("name", map_names())
def test_load_all_maps(name):
    tunnel_map = TunnelMap.from_name(name)
    assert tunnel_map.walls[tunnel_map.start] == 0
    assert tunnel_map.diameter > 0


def test_specs():
    maps = [
        TunnelMap.from_name("old_envs/map1.csv"),
        TunnelMap.from_name("old_envs/env_cross.csv"),
    ]
    check_env_specs(TunnelEnv(maps, num_envs=3, n_agents=2, seed=0))


def test_walls_block_moves(small_map):
    env = TunnelEnv(small_map, n_agents=1)
    env.reset()
    act(env, [[1]])
    assert env.agent_positions()[0, 0].tolist() == [1, 2]


def test_open_border_blocks_moves():
    open_map = TunnelMap(
        walls=np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int8), start=(0, 0), diameter=3
    )
    env = TunnelEnv(open_map, n_agents=1)
    env.reset()
    act(env, [[3]])
    act(env, [[1]])
    assert env.agent_positions()[0, 0].tolist() == [0, 0]


def corridor_env(n_agents):
    corridor = TunnelMap(
        walls=np.zeros((1, 5), dtype=np.int8), start=(0, 0), diameter=4
    )
    env = TunnelEnv(corridor, n_agents=n_agents)
    env.reset()
    return env


def cols_of(env):
    return env.agent_positions()[0, :, 1].tolist()


def test_leaving_the_start_one_at_a_time():
    env = corridor_env(2)
    act(env, [[4, 4]])
    assert sorted(cols_of(env)) == [0, 1]


def test_conflict_winner_is_random():
    corridor = TunnelMap(
        walls=np.zeros((1, 5), dtype=np.int8), start=(0, 0), diameter=4
    )
    env = TunnelEnv(corridor, num_envs=200, n_agents=2, seed=0)
    env.reset()
    act(env, [[4, 4]] * 200)
    cols = env.agent_positions()[..., 1]
    assert (cols.sum(-1) == 1).all()
    assert 50 < int((cols[:, 0] == 1).sum()) < 150


def test_drones_follow_each_other():
    env = corridor_env(2)
    act(env, [[0, 4]])
    act(env, [[4, 4]])
    assert cols_of(env) == [1, 2]


def test_blocked_by_still_drone():
    env = corridor_env(2)
    act(env, [[0, 4]])
    act(env, [[4, 0]])
    assert cols_of(env) == [0, 1]


def test_drones_cannot_swap():
    env = corridor_env(2)
    act(env, [[0, 4]])
    act(env, [[4, 3]])
    assert cols_of(env) == [0, 1]


def test_cannot_go_back_to_occupied_start():
    env = corridor_env(3)
    act(env, [[0, 0, 4]])
    act(env, [[0, 0, 3]])
    assert cols_of(env) == [0, 0, 1]


def test_grouped_reward_counts_new_cells(small_map):
    env = TunnelEnv(small_map, n_agents=2, step_penalty=0.0)
    env.reset()
    out = act(env, [[3, 4]])
    assert rewards_of(out) == [2.0, 2.0]
    out = act(env, [[4, 3]])
    assert rewards_of(out) == [0.0, 0.0]
    assert env.agent_positions()[0].tolist() in (
        [[1, 2], [1, 3]],
        [[1, 1], [1, 2]],
    )


def test_mixed_reward(small_map):
    env = TunnelEnv(
        small_map, n_agents=2, step_penalty=0.0, reward_mode="mixed", reward_alpha=0.5
    )
    env.reset()
    out = act(env, [[3, 0]])
    assert rewards_of(out) == [1.0, 0.5]


def test_ray_observation(small_map):
    env = TunnelEnv(small_map, n_agents=2, view_range=2)
    obs = env.reset()[GROUP, "observation"][0, 0]
    assert obs.shape == (4 * 2 * 3 + 1,)
    rays = rays_of(obs, 2)
    assert rays[0].tolist() == [[1, 0, 0], [1, 0, 0]]
    assert rays[1].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert rays[2].tolist() == [[0, 0, 0], [1, 0, 0]]
    assert rays[3].tolist() == [[0, 0, 0], [1, 0, 0]]
    assert obs[-1] == 1.0

    obs = act(env, [[0, 2]])[GROUP, "observation"][0, 0]
    assert rays_of(obs, 2)[1].tolist() == [[0, 1, 1], [0, 0, 0]]
    assert obs[-1] == 0.0


def test_vision_blocked_by_walls():
    corridor = TunnelMap(
        walls=np.array([[0, 1, 0, 0]], dtype=np.int8), start=(0, 0), diameter=0
    )
    env = TunnelEnv(corridor, n_agents=1, view_range=3)
    obs = env.reset()[GROUP, "observation"][0, 0]
    assert rays_of(obs, 3)[3, :, 0].tolist() == [1, 1, 1]


def test_terminates_when_fully_explored(small_map):
    env = TunnelEnv(small_map, n_agents=1)
    env.reset()
    path = [3, 2, 2, 4, 1, 4, 1, 2, 2]
    for i, action in enumerate(path):
        out = act(env, [[action]])
        assert out["terminated"].item() == (i == len(path) - 1)
    assert out["done"].item()
    assert out[GROUP, "info", "coverage"][0, 0] == 1.0


def test_truncation(small_map):
    env = TunnelEnv(small_map, n_agents=1, max_steps=3)
    env.reset()
    for _ in range(3):
        out = act(env, [[0]])
    assert out["truncated"].item()
    assert not out["terminated"].item()


def test_truncates_when_discovery_stalls(small_map):
    env = TunnelEnv(small_map, n_agents=1)
    env.reset()
    for _ in range(3):
        out = act(env, [[0]])
    assert not out["truncated"].item()
    out = act(env, [[0]])
    assert out["truncated"].item()
    assert not out["terminated"].item()


def test_discovery_resets_stall_counter(small_map):
    env = TunnelEnv(small_map, n_agents=1)
    env.reset()
    for action in [0, 0, 0, 3, 0, 0, 0]:
        out = act(env, [[action]])
    assert not out["truncated"].item()


def test_partial_reset(small_map):
    env = TunnelEnv(small_map, num_envs=2, n_agents=1)
    env.reset()
    act(env, [[2], [2]])
    env.reset(TensorDict({"_reset": torch.tensor([[True], [False]])}, batch_size=[2]))
    positions = env.agent_positions()[:, 0].tolist()
    assert positions == [[1, 2], [2, 2]]
    assert env.coverage().tolist() == pytest.approx([1 / 9, 2 / 9])


def test_envs_are_independent():
    tunnel_map = TunnelMap.from_name("old_envs/map2.csv")
    batched = TunnelEnv(tunnel_map, num_envs=3, n_agents=1, seed=0)
    singles = [TunnelEnv(tunnel_map, n_agents=1, seed=0) for _ in range(3)]
    batched.reset()
    for env in singles:
        env.reset()

    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        actions = torch.randint(5, (3, 1), generator=generator)
        out = act(batched, actions.tolist())
        for i, env in enumerate(singles):
            single = act(env, actions[i : i + 1].tolist())
            assert (out[GROUP][i] == single[GROUP][0]).all()


def test_rollout_auto_resets():
    env = TunnelEnv(
        TunnelMap.from_name("old_envs/map1.csv"), num_envs=4, max_steps=20, seed=0
    )
    rollout = env.rollout(100, break_when_any_done=False)
    assert rollout.batch_size == (4, 100)
    assert rollout["next", "truncated"].sum() == 4 * 5
    assert int(env.n_steps.max()) <= 20


def test_hot_vector(small_map):
    env = TunnelEnv(small_map, num_envs=2, n_agents=3, view_range=2, hot_vector=True)
    check_env_specs(env)
    obs = env.reset()[GROUP, "observation"]
    rays_size = 4 * 2 * 3
    assert obs.shape == (2, 3, rays_size + 1 + 3)
    assert obs[:, :, rays_size + 1 :].tolist() == [torch.eye(3).tolist()] * 2

    plain = TunnelEnv(small_map, num_envs=2, n_agents=3, view_range=2)
    plain_obs = plain.reset()[GROUP, "observation"]
    assert torch.equal(obs[..., : rays_size + 1], plain_obs)
