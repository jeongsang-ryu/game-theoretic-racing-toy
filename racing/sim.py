"""1v1 racing toy simulator in Frenet coordinates (vectorized with numpy).

State per car: s (progress along centerline, unwrapped), d (lateral offset, + = left/inside), v (speed).
Track: oval, two straights + two left-hand 180 deg corners.
Interaction: collision (both lose speed) and slipstream (follower right behind gets higher top speed).
"""
import numpy as np

# ---------------- track ----------------
STRAIGHT = 120.0
RADIUS = 25.0
CORNER = np.pi * RADIUS
L = 2 * STRAIGHT + 2 * CORNER          # lap length (~317 m)
D_MAX = 4.5                             # |d| limit (half width 5 m minus car half width)

# ---------------- car / physics ----------------
DT = 0.1
V_TOP = np.array([30.0, 32.5])   # car1 (attacker) has a slightly faster car
A_MAX = 4.0
B_MAX = 8.0
A_LAT = 12.0
VLAT_MAX = 3.0
KD = 1.5
KV = 3.0
CAR_LEN = 4.5
CAR_WID = 1.8
SLIP_RANGE = (3.0, 20.0)
SLIP_DD = 2.0
SLIP_BONUS_V = 2.0
SLIP_BONUS_A = 1.0
COLL_SPEED_FACTOR = 0.6

# ---------------- action set: motion primitives ----------------
D_TARGETS = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
SPEED_MODES = np.array([1.0, 0.85])
PRIMS = np.array([(dt, sm) for sm in SPEED_MODES for dt in D_TARGETS])  # K x 2
K = len(PRIMS)


def kappa(s):
    s = np.mod(s, L)
    c1 = (s >= STRAIGHT) & (s < STRAIGHT + CORNER)
    c2 = s >= 2 * STRAIGHT + CORNER
    return np.where(c1 | c2, 1.0 / RADIUS, 0.0)


def v_limit(s, d):
    k = kappa(s)
    reff = np.where(k > 0, 1.0 / np.maximum(k, 1e-9) - d, 1e9)
    return np.minimum(np.sqrt(A_LAT * np.maximum(reff, 1.0)), 60.0)


LOOK = np.arange(0.0, 65.0, 5.0)


def v_allow(s, d_tgt):
    """max speed now such that we can brake for upcoming corner (assuming lateral target d_tgt)."""
    ss = s[..., None] + LOOK
    vl = v_limit(ss, d_tgt[..., None])
    return np.min(np.sqrt(vl ** 2 + 2 * B_MAX * LOOK), axis=-1)


def wrap(x):
    return (x + L / 2) % L - L / 2


def xy(s, d):
    """map Frenet (s,d) to cartesian for plotting. Track centered at origin."""
    s = np.mod(s, L)
    x = np.zeros_like(s); y = np.zeros_like(s); hx = np.zeros_like(s); hy = np.zeros_like(s)
    # segment 1: bottom straight, going +x from (-40,-25)
    m = s < STRAIGHT
    x[m] = -STRAIGHT / 2 + s[m]; y[m] = -RADIUS; hx[m] = 1; hy[m] = 0
    # corner 1: right end, center (40,0)
    m = (s >= STRAIGHT) & (s < STRAIGHT + CORNER)
    th = (s[m] - STRAIGHT) / RADIUS - np.pi / 2
    x[m] = STRAIGHT / 2 + RADIUS * np.cos(th); y[m] = RADIUS * np.sin(th)
    hx[m] = -np.sin(th); hy[m] = np.cos(th)
    # straight 2: top, going -x
    m = (s >= STRAIGHT + CORNER) & (s < 2 * STRAIGHT + CORNER)
    u = s[m] - STRAIGHT - CORNER
    x[m] = STRAIGHT / 2 - u; y[m] = RADIUS; hx[m] = -1; hy[m] = 0
    # corner 2: left end, center (-40,0)
    m = s >= 2 * STRAIGHT + CORNER
    th = (s[m] - 2 * STRAIGHT - CORNER) / RADIUS + np.pi / 2
    x[m] = -STRAIGHT / 2 + RADIUS * np.cos(th); y[m] = RADIUS * np.sin(th)
    hx[m] = -np.sin(th); hy[m] = np.cos(th)
    # left normal = (-hy, hx)
    return x + d * (-hy), y + d * hx


def step(S, P, contact_prev):
    """One sim step for both cars.
    S: dict with arrays s,d,v of shape (..., 2)
    P: primitive indices, shape (..., 2)
    contact_prev: bool (...,) whether cars were in contact last step
    returns new S, collision_event (...,), contact (...,)
    """
    s, d, v = S["s"], S["d"], S["v"]
    dtg = PRIMS[P, 0]
    mode = PRIMS[P, 1]
    # slipstream: car i is right behind car j in same lane
    ds_ji = wrap(s[..., ::-1] - s)          # other minus me
    dd = np.abs(d[..., ::-1] - d)
    slip = (ds_ji > SLIP_RANGE[0]) & (ds_ji < SLIP_RANGE[1]) & (dd < SLIP_DD)
    vtop = V_TOP + SLIP_BONUS_V * slip
    amax = A_MAX + SLIP_BONUS_A * slip
    # low-level controller
    ud = np.clip(KD * (dtg - d), -VLAT_MAX, VLAT_MAX)
    vref = np.minimum(mode * vtop, v_allow(s, dtg))
    a = np.clip(KV * (vref - v), -B_MAX, amax)
    v_new = np.clip(v + a * DT, 0.0, None)
    d_new = np.clip(d + ud * DT, -D_MAX, D_MAX)
    k = kappa(s)
    s_new = s + v * DT / np.maximum(1.0 - k * d, 0.3)
    # grip limit (should rarely bind)
    v_new = np.minimum(v_new, v_limit(s_new, d_new) + 0.5)
    # collision: lateral & longitudinal overlap
    ds = wrap(s_new[..., 1] - s_new[..., 0])
    ddd = d_new[..., 1] - d_new[..., 0]
    contact = (np.abs(ds) < CAR_LEN) & (np.abs(ddd) < CAR_WID)
    event = contact & ~contact_prev
    v_new = np.where(event[..., None], v_new * COLL_SPEED_FACTOR, v_new)
    # resolve overlap: rear car is placed just behind the front car and cannot be faster than it
    s_new = s_new.copy(); v_new = v_new.copy()
    rear1 = contact & (ds >= 0)          # car0 is behind car1
    rear0 = contact & (ds < 0)           # car1 is behind car0
    s_new[..., 0] = np.where(rear1, s_new[..., 0] - (CAR_LEN - ds), s_new[..., 0])
    s_new[..., 1] = np.where(rear0, s_new[..., 1] - (CAR_LEN + ds), s_new[..., 1])
    v_new[..., 0] = np.where(rear1, np.minimum(v_new[..., 0], v_new[..., 1]), v_new[..., 0])
    v_new[..., 1] = np.where(rear0, np.minimum(v_new[..., 1], v_new[..., 0]), v_new[..., 1])
    contact = contact | (contact_prev & (np.abs(ds) < CAR_LEN + 1.0) & (np.abs(ddd) < CAR_WID + 0.5))
    return {"s": s_new, "d": d_new, "v": v_new}, event, contact


def rollout_matrix(S0, contact0, me, H=20, solo=False, opp_fixed=None):
    """Evaluate all K x K joint primitive pairs over H steps.
    S0: dict of scalars arrays shape (2,), me: index of planning car.
    Returns prog (K,K,2) progress gains and coll (K,K) collision indicator.
    rows = my primitive, cols = opponent primitive.
    solo=True: opponent is removed (placed far away).
    """
    op = 1 - me
    ia, ib = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    P = np.zeros((K, K, 2), dtype=int)
    P[..., me] = ia
    P[..., op] = ib
    S = {k: np.broadcast_to(S0[k], (K, K, 2)).copy() for k in S0}
    if solo:
        S["s"][..., op] = S["s"][..., op] + L / 2  # far away
        S["d"][..., op] = 0.0
    c = np.broadcast_to(contact0, (K, K)).copy() if not solo else np.zeros((K, K), bool)
    s_start = S["s"].copy()
    coll = np.zeros((K, K))
    for _ in range(H):
        S, ev, c = step(S, P, c)
        coll += ev + 0.1 * c          # new collision + time spent in contact
    prog = S["s"] - s_start
    return prog, coll


def payoff(prog, coll, me, w_own=0.2, w_coll=30.0):
    op = 1 - me
    return (prog[..., me] - prog[..., op]) + w_own * prog[..., me] - w_coll * coll
