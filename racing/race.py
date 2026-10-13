import numpy as np
from . import sim

T_RACE = 35.0        # seconds (~2 laps)
DECIDE_EVERY = 2     # sim steps between decisions (0.2 s)


def init_state(rng, gap=None):
    """car 0 = leader (defender), car 1 = follower (attacker)."""
    s0 = rng.uniform(0, 20)            # start on the bottom straight
    g = rng.uniform(6, 12) if gap is None else gap
    d0 = rng.choice([-2.0, 0.0, 2.0])
    d1 = rng.choice([-2.0, 0.0, 2.0])
    v = 22.0
    return {"s": np.array([s0 + g, s0]), "d": np.array([d0, d1]), "v": np.array([v, v])}


def run_race(agents, S, rng, T=T_RACE, log=False):
    """agents: [agent_for_car0, agent_for_car1]"""
    for a in agents:
        a.reset(rng)
    contact = np.array(False)
    P = np.zeros(2, dtype=int)
    n = int(T / sim.DT)
    traj = []
    collisions = 0
    lead_changes = 0
    leader = int(np.argmax(S["s"]))
    for k in range(n):
        if k % DECIDE_EVERY == 0:
            P = np.array([agents[0].act(S, contact, 0), agents[1].act(S, contact, 1)])
        S, ev, contact = sim.step(S, P, contact)
        collisions += int(ev)
        new_leader = int(np.argmax(S["s"]))
        if new_leader != leader:
            lead_changes += 1
            leader = new_leader
        if log:
            traj.append((S["s"].copy(), S["d"].copy(), S["v"].copy(), P.copy(), bool(ev)))
    gap = S["s"][0] - S["s"][1]   # >0: car0 (defender) ahead
    return {"gap": gap, "winner": 0 if gap > 0 else 1, "collisions": collisions,
            "lead_changes": lead_changes, "traj": traj, "dist": S["s"].copy()}
