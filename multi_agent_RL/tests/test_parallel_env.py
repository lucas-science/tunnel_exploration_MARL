import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from multi_agent_RL import TunnelExplorationEnv, TunnelMap
from multi_agent_RL.tunnel_map import MAPS_DIR

SMALL_MAP_CSV = """\
2,1,,,
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


def test_load_map(small_map):
    assert small_map.shape == (5, 5)
    assert small_map.start == (1, 2)
    assert small_map.n_tunnels == 9


@pytest.mark.parametrize("name", sorted(p.name for p in MAPS_DIR.glob("*.csv")))
def test_load_all_maps(name):
    tunnel_map = TunnelMap.from_name(name)
    assert tunnel_map.walls[tunnel_map.start] == 0


def test_parallel_api():
    env = TunnelExplorationEnv(TunnelMap.from_name("map2.csv"), max_steps=50)
    parallel_api_test(env, num_cycles=100)


def test_walls_block_moves(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=1)
    env.reset(seed=0)
    env.step({"drone_0": 1})
    assert tuple(env.positions[0]) == (1, 2)


def test_shared_reward_counts_new_cells(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=2, step_penalty=0.0)
    env.reset(seed=0)
    _, rewards, *_ = env.step({"drone_0": 3, "drone_1": 4})
    assert rewards == {"drone_0": 2.0, "drone_1": 2.0}
    _, rewards, *_ = env.step({"drone_0": 4, "drone_1": 3})
    assert rewards == {"drone_0": 0.0, "drone_1": 0.0}


def test_ray_observation(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=2, view_range=2)
    obs, _ = env.reset(seed=0)
    assert obs["drone_0"].shape == (4 * 2 * 3 + 1,)
    rays = obs["drone_0"][:-1].reshape(4, 2, 3)
    np.testing.assert_array_equal(rays[0], [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(rays[1], [[0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(rays[2], [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(rays[3], [[0, 0, 0], [1, 0, 0]])
    assert obs["drone_0"][-1] == 1.0

    obs, *_ = env.step({"drone_0": 0, "drone_1": 2})
    rays = obs["drone_0"][:-1].reshape(4, 2, 3)
    np.testing.assert_array_equal(rays[1], [[0, 1, 1], [0, 0, 0]])
    assert obs["drone_0"][-1] == 0.0


def test_vision_blocked_by_walls():
    corridor = TunnelMap(walls=np.array([[0, 1, 0, 0]], dtype=np.int8), start=(0, 0))
    env = TunnelExplorationEnv(corridor, n_agents=1, view_range=3)
    obs, _ = env.reset(seed=0)
    rays = obs["drone_0"][:-1].reshape(4, 3, 3)
    np.testing.assert_array_equal(rays[3, :, 0], [1, 1, 1])


def test_terminates_when_fully_explored(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=1)
    env.reset(seed=0)
    path = [3, 2, 2, 4, 1, 4, 1, 2, 2]
    for i, action in enumerate(path):
        _, _, terminations, _, _ = env.step({"drone_0": action})
        if i < len(path) - 1:
            assert not terminations["drone_0"]
    assert terminations["drone_0"]
    assert env.agents == []


def test_truncation(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=1, max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, _, truncations, _ = env.step({"drone_0": 0})
    assert truncations["drone_0"]
    assert env.agents == []


def test_mixed_reward(small_map):
    env = TunnelExplorationEnv(
        small_map, n_agents=2, step_penalty=0.0, reward_alpha=0.5
    )
    env.reset(seed=0)
    rewards = env._mixed_reward({"drone_0": 3, "drone_1": 0})
    assert rewards == {"drone_0": 1.0, "drone_1": 0.5}


def test_open_border_blocks_moves():
    open_map = TunnelMap(
        walls=np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int8), start=(0, 0)
    )
    env = TunnelExplorationEnv(open_map, n_agents=1)
    env.reset(seed=0)
    env.step({"drone_0": 3})
    env.step({"drone_0": 1})
    assert tuple(env.positions[0]) == (0, 0)


def test_missing_action_stays(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=2)
    env.reset(seed=0)
    env.step({"drone_0": 3})
    assert tuple(env.positions[1]) == (1, 2)


def test_step_after_end_does_not_crash(small_map):
    env = TunnelExplorationEnv(small_map, n_agents=1, max_steps=1)
    env.reset(seed=0)
    env.step({"drone_0": 0})
    _, rewards, *_ = env.step({})
    assert rewards == {}
