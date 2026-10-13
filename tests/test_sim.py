import numpy as np
from racing import sim, agents, race


def state(s0, s1, d0=0.0, d1=0.0, v=20.0):
    return {"s": np.array([s0, s1]), "d": np.array([d0, d1]), "v": np.array([v, v])}


def test_primitives():
    assert sim.K == 10 and sim.PRIMS.shape == (10, 2)


def test_corner_speed_limit_inside_is_slower():
    s_corner = sim.STRAIGHT + 10.0
    assert sim.v_limit(np.array(s_corner), np.array(4.0)) < sim.v_limit(np.array(s_corner), np.array(-4.0))


def test_collision_event_and_separation():
    S = state(10.0, 8.0)  # car1 2 m behind car0 in the same lane -> overlap
    S2, ev, contact = sim.step(S, np.array([0, 0]), np.array(False))
    assert bool(ev) and bool(contact)
    assert S2["s"][0] - S2["s"][1] >= sim.CAR_LEN - 1e-6   # rear car pushed behind


def test_no_interaction_when_far_apart():
    S = state(0.0, sim.L / 2)
    _, ev, contact = sim.step(S, np.array([2, 2]), np.array(False))
    assert not ev and not contact


def test_rollout_matrix_shapes():
    prog, coll = sim.rollout_matrix(state(20.0, 10.0), np.array(False), me=0, H=10)
    assert prog.shape == (sim.K, sim.K, 2) and coll.shape == (sim.K, sim.K)
    assert (prog >= 0).all()


def test_minimax_strategy_is_a_distribution():
    a = agents.Minimax(); a.reset(np.random.default_rng(0))
    act = a.act(state(20.0, 12.0), np.array(False), 0)
    assert 0 <= act < sim.K


def test_every_planner_returns_valid_action():
    S = state(20.0, 12.0, 2.0, -2.0)
    for A in [agents.Solo, agents.Predictive, agents.Nash, agents.Stackelberg]:
        a = A(); a.reset(np.random.default_rng(0))
        assert 0 <= a.act(S, np.array(False), 1) < sim.K


def test_short_race_runs():
    rng = np.random.default_rng(0)
    r = race.run_race([agents.Solo(), agents.Predictive()], race.init_state(rng), rng, T=3.0)
    assert r["winner"] in (0, 1)
