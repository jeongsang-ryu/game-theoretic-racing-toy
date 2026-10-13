import numpy as np
from racing import rl, sim, race


def test_observation_shape():
    rng = np.random.default_rng(0)
    S = rl.reset_batch(rng, 4)
    o = rl.observe(S, np.zeros(4, bool), 0)
    assert o.shape == (4, rl.OBS_DIM) and np.isfinite(o).all()


def test_pretrained_policy_acts():
    a = rl.RLAgent(); a.reset(np.random.default_rng(0))
    S = race.init_state(np.random.default_rng(1))
    assert 0 <= a.act(S, np.array(False), 0) < sim.K
