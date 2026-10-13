"""Planning agents. Each returns a primitive index given the joint state."""
import numpy as np
from scipy.optimize import linprog
from . import sim

H = 20  # planning horizon steps (2.0 s)


class Agent:
    name = "base"

    def reset(self, rng):
        self.rng = rng

    def act(self, S, contact, me):
        raise NotImplementedError


class Solo(Agent):
    """(1) single-agent MPC: maximize own progress, ignore opponent."""
    name = "Solo MPC"

    def act(self, S, contact, me):
        prog, coll = sim.rollout_matrix(S, contact, me, H, solo=True)
        own = prog[:, 0, me]
        return int(np.argmax(own))


def hold_prim(S, who):
    """primitive closest to 'keep current lane at full speed'"""
    j = int(np.argmin(np.abs(sim.D_TARGETS - S["d"][who])))
    return j  # speed mode 1.0 occupies first len(D_TARGETS) entries


class Predictive(Agent):
    """(2) predict opponent keeps lane & speed, best-respond to that prediction."""
    name = "Predict+MPC"

    def act(self, S, contact, me):
        prog, coll = sim.rollout_matrix(S, contact, me, H)
        U = sim.payoff(prog, coll, me)
        b = hold_prim(S, 1 - me)
        return int(np.argmax(U[:, b]))


def ibr(Ui, Uj, a0, b0, iters=10):
    a, b = a0, b0
    seen = set()
    for _ in range(iters):
        a_new = int(np.argmax(Ui[:, b]))
        b_new = int(np.argmax(Uj[a_new, :]))
        if (a_new, b_new) == (a, b):
            return a, b, True
        if (a_new, b_new) in seen:
            break
        seen.add((a_new, b_new))
        a, b = a_new, b_new
    return a, b, False


class Nash(Agent):
    """(3a) game-theoretic: iterated best response on the bimatrix game -> pure Nash equilibrium."""
    name = "Nash (IBR)"

    def act(self, S, contact, me):
        prog, coll = sim.rollout_matrix(S, contact, me, H)
        Ui = sim.payoff(prog, coll, me)
        Uj = sim.payoff(prog, coll, 1 - me)
        a0 = int(np.argmax(Ui[:, hold_prim(S, 1 - me)]))
        b0 = hold_prim(S, 1 - me)
        a, b, ok = ibr(Ui, Uj, a0, b0)
        if not ok:  # no pure equilibrium found -> fall back to maximin (security) strategy
            a = int(np.argmax(Ui.min(axis=1)))
        return a


class Stackelberg(Agent):
    """(3b) leader: choose a anticipating opponent's best response BR(a)."""
    name = "Stackelberg"

    def act(self, S, contact, me):
        prog, coll = sim.rollout_matrix(S, contact, me, H)
        Ui = sim.payoff(prog, coll, me)
        Uj = sim.payoff(prog, coll, 1 - me)
        br = np.argmax(Uj, axis=1)                  # follower response to each leader action
        val = Ui[np.arange(sim.K), br]
        return int(np.argmax(val))


class Minimax(Agent):
    """(3c) zero-sum security strategy via LP:  max_x min_b x^T U[:,b]  (von Neumann)."""
    name = "Minimax (LP)"

    def act(self, S, contact, me):
        prog, coll = sim.rollout_matrix(S, contact, me, H)
        U = sim.payoff(prog, coll, me)
        Kn = sim.K
        # variables [x (K), t]; maximize t  -> minimize -t
        c = np.zeros(Kn + 1); c[-1] = -1
        A_ub = np.hstack([-U.T, np.ones((Kn, 1))])   # t - x^T U[:,b] <= 0
        b_ub = np.zeros(Kn)
        A_eq = np.hstack([np.ones((1, Kn)), np.zeros((1, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * Kn + [(None, None)], method="highs")
        x = np.clip(res.x[:Kn], 0, None)
        x = x / x.sum()
        return int(self.rng.choice(Kn, p=x))
