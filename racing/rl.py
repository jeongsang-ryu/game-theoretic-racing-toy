"""(4) Deep RL: PPO with self-play (+ league of past snapshots), implemented in JAX.

Markov game: o_i -> pi_theta(a_i | o_i), a_i in the same K motion primitives.
Both cars share one policy; a role flag tells the policy which car it drives.
"""
import time, pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from . import sim

DECIDE_EVERY = 2
T_EP = 35.0
N_DEC = int(T_EP / sim.DT / DECIDE_EVERY)   # decisions per episode
LOOK_K = np.array([0.0, 15.0, 30.0, 45.0, 60.0])
OBS_DIM = 2 + 4 + len(LOOK_K) + 2


def observe(S, contact, me):
    """S arrays (B,2). returns obs (B, OBS_DIM) for car `me`."""
    op = 1 - me
    s, d, v = S["s"], S["d"], S["v"]
    ds = np.clip(sim.wrap(s[:, op] - s[:, me]) / 20.0, -3, 3)
    feats = [d[:, me] / 4.5, v[:, me] / 30.0,
             ds, (d[:, op] - d[:, me]) / 4.5, v[:, op] / 30.0, d[:, op] / 4.5]
    curv = sim.kappa(s[:, me][:, None] + LOOK_K) * sim.RADIUS
    role = np.full(len(s), float(me))
    return np.concatenate([np.stack(feats, 1), curv, role[:, None], contact[:, None].astype(float)], 1).astype(np.float32)


# ---------------- network ----------------
def init_params(key, hid=128):
    def lin(k, i, o, scale=1.0):
        return {"w": jax.random.normal(k, (i, o)) * scale / np.sqrt(i), "b": jnp.zeros(o)}
    ks = jax.random.split(key, 6)
    return {"p1": lin(ks[0], OBS_DIM, hid), "p2": lin(ks[1], hid, hid), "pi": lin(ks[2], hid, sim.K, 0.01),
            "v1": lin(ks[3], OBS_DIM, hid), "v2": lin(ks[4], hid, hid), "vo": lin(ks[5], hid, 1, 1.0)}


def forward(p, x):
    h = jnp.tanh(x @ p["p1"]["w"] + p["p1"]["b"])
    h = jnp.tanh(h @ p["p2"]["w"] + p["p2"]["b"])
    logits = h @ p["pi"]["w"] + p["pi"]["b"]
    g = jnp.tanh(x @ p["v1"]["w"] + p["v1"]["b"])
    g = jnp.tanh(g @ p["v2"]["w"] + p["v2"]["b"])
    return logits, (g @ p["vo"]["w"] + p["vo"]["b"])[..., 0]


forward_j = jax.jit(forward)


@jax.jit
def sample(p, x, key):
    logits, v = forward(p, x)
    a = jax.random.categorical(key, logits)
    logp = jax.nn.log_softmax(logits)[jnp.arange(x.shape[0]), a]
    return a, logp, v


def ppo_loss(p, obs, act, oldlogp, adv, ret, clip=0.2, vcoef=0.5, ent=0.005):
    logits, v = forward(p, obs)
    lsm = jax.nn.log_softmax(logits)
    logp = lsm[jnp.arange(obs.shape[0]), act]
    ratio = jnp.exp(logp - oldlogp)
    pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv))
    vl = jnp.mean((v - ret) ** 2)
    entropy = -jnp.mean(jnp.sum(jnp.exp(lsm) * lsm, -1))
    return pg + vcoef * vl - ent * entropy, (pg, vl, entropy)


# simple Adam
def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": z, "t": 0}


@jax.jit
def update(p, opt_m, opt_v, t, obs, act, oldlogp, adv, ret, lr):
    (loss, aux), g = jax.value_and_grad(ppo_loss, has_aux=True)(p, obs, act, oldlogp, adv, ret)
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, 0.5 / (gn + 1e-8)), g)
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt_m, g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, opt_v, g)
    mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    p = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + 1e-8), p, mh, vh)
    return p, m, v, loss, aux


DEFAULT_MODEL_OUT = Path(__file__).resolve().parents[1] / "models" / "ppo_selfplay.pkl"


# ---------------- environment rollout ----------------
def reset_batch(rng, B):
    s0 = rng.uniform(0, sim.L, B)
    g = rng.uniform(6, 12, B)
    d = rng.choice([-2.0, 0.0, 2.0], (B, 2))
    v = np.full((B, 2), 22.0)
    return {"s": np.stack([s0 + g, s0], 1), "d": d, "v": v}


def rollout(params, opp_params, rng, key, B):
    """Car roles: car0 & car1. In half the envs the learner drives car0, in the other half car1;
    opponent uses opp_params (current params => pure self-play)."""
    S = reset_batch(rng, B)
    contact = np.zeros(B, bool)
    learner_car = (np.arange(B) % 2).astype(int)
    buf = {k: [] for k in ["obs", "act", "logp", "val", "rew", "done"]}
    for t in range(N_DEC):
        o0 = observe(S, contact, 0); o1 = observe(S, contact, 1)
        oL = np.where(learner_car[:, None] == 0, o0, o1)
        oO = np.where(learner_car[:, None] == 0, o1, o0)
        key, k1, k2 = jax.random.split(key, 3)
        aL, lpL, vL = sample(params, jnp.asarray(oL), k1)
        aO, _, _ = sample(opp_params, jnp.asarray(oO), k2)
        aL = np.asarray(aL); aO = np.asarray(aO)
        P = np.where(learner_car[:, None] == 0, np.stack([aL, aO], 1), np.stack([aO, aL], 1))
        s_before = S["s"].copy()
        ev_sum = np.zeros(B); c_sum = np.zeros(B)
        for _ in range(DECIDE_EVERY):
            S, ev, contact = sim.step(S, P, contact)
            ev_sum += ev; c_sum += contact
        prog = S["s"] - s_before
        me = learner_car; op = 1 - learner_car
        pm = prog[np.arange(B), me]; po = prog[np.arange(B), op]
        r = (pm - po) / 5.0 + 0.05 * pm / 5.0 - 3.0 * ev_sum - 0.3 * c_sum
        done = np.zeros(B)
        if t == N_DEC - 1:
            ahead = sim.wrap(S["s"][np.arange(B), me] - S["s"][np.arange(B), op]) > 0
            r = r + np.where(ahead, 3.0, -3.0)
            done[:] = 1
        buf["obs"].append(oL); buf["act"].append(aL); buf["logp"].append(np.asarray(lpL))
        buf["val"].append(np.asarray(vL)); buf["rew"].append(r); buf["done"].append(done)
    return {k: np.array(v) for k, v in buf.items()}, key


def gae(buf, gamma=0.995, lam=0.95):
    T, B = buf["rew"].shape
    adv = np.zeros((T, B)); last = np.zeros(B)
    for t in reversed(range(T)):
        nv = buf["val"][t + 1] if t + 1 < T else np.zeros(B)
        nd = 1 - buf["done"][t]
        delta = buf["rew"][t] + gamma * nv * nd - buf["val"][t]
        last = delta + gamma * lam * nd * last
        adv[t] = last
    return adv, adv + buf["val"]


def train(iters=1000, B=128, seed=0, out=DEFAULT_MODEL_OUT):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    params = init_params(k)
    opt = adam_init(params); m, v, t = opt["m"], opt["v"], 0
    league = []
    log = []
    t0 = time.time()
    for it in range(iters):
        # opponent: current self (60%) or a past snapshot (40%)
        if league and rng.random() < 0.4:
            opp = league[rng.integers(len(league))]
        else:
            opp = params
        buf, key = rollout(params, opp, rng, key, B)
        adv, ret = gae(buf)
        N = adv.size
        obs = buf["obs"].reshape(N, -1); act = buf["act"].reshape(N); lp = buf["logp"].reshape(N)
        adv_f = adv.reshape(N); ret_f = ret.reshape(N)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        lr = 3e-4 * (1 - it / iters) + 3e-5
        for ep in range(4):
            perm = rng.permutation(N)
            for mb in np.array_split(perm, 4):
                t += 1
                params, m, v, loss, aux = update(params, m, v, t, jnp.asarray(obs[mb]), jnp.asarray(act[mb]),
                                                 jnp.asarray(lp[mb]), jnp.asarray(adv_f[mb]), jnp.asarray(ret_f[mb]), lr)
        if it % 20 == 0:
            league.append(jax.tree_util.tree_map(lambda x: x, params))
        ep_ret = buf["rew"].sum(0).mean()
        log.append((it, float(ep_ret), float(aux[2]), time.time() - t0))
        if it % 10 == 0:
            print(f"it {it:4d} ret {ep_ret:7.2f} ent {float(aux[2]):.3f} t {time.time()-t0:6.1f}s", flush=True)
    with open(out, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "log": log}, f)
    return params


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "ppo_selfplay.pkl"
_CACHE = {}


def load_params(path=DEFAULT_MODEL):
    path = str(path)
    if path not in _CACHE:
        with open(path, "rb") as f:
            _CACHE[path] = jax.tree_util.tree_map(jnp.asarray, pickle.load(f)["params"])
    return _CACHE[path]


class RLAgent:
    name = "Deep RL (self-play)"

    def __init__(self, path=DEFAULT_MODEL):
        self.params = load_params(path)

    def reset(self, rng):
        self.rng = rng

    def act(self, S, contact, me):
        S2 = {k: v[None] for k, v in S.items()}
        o = observe(S2, np.array([bool(contact)]), me)
        logits, _ = forward_j(self.params, jnp.asarray(o))
        return int(np.argmax(np.asarray(logits)[0]))

