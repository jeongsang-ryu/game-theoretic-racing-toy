"""Round-robin 1v1 tournament: every method defends (car 0: starts ahead, slower car)
against every method attacking (car 1: starts behind, faster car)."""
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from . import agents, race, rl

METHODS = [agents.Solo, agents.Predictive, agents.Nash, agents.Stackelberg, agents.Minimax, rl.RLAgent]
NAMES = [m.name for m in METHODS]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "results.json"


def play(args):
    """One race. args = (defender index, attacker index, seed)."""
    i, j, seed = args
    rng = np.random.default_rng(seed)
    S = race.init_state(rng)
    r = race.run_race([METHODS[i](), METHODS[j]()], S, np.random.default_rng(seed + 1000))
    return i, j, seed, int(r["winner"] == 0), r["collisions"], float(r["gap"]), r["lead_changes"]


def run(n_races=16, workers=2, methods=None, out=DEFAULT_OUT):
    idx = range(len(METHODS)) if methods is None else methods
    jobs = [(i, j, s) for i in idx for j in idx for s in range(n_races)]
    if workers > 1:
        with Pool(workers) as p:
            res = p.map(play, jobs, chunksize=4)
    else:
        res = [play(job) for job in jobs]
    if out is not None:
        json.dump({"names": NAMES, "N": n_races, "results": res}, open(out, "w"))
    return res


def summarize(res, n_races):
    n = len(METHODS)
    W = np.zeros((n, n)); C = np.zeros((n, n))
    for i, j, s, w, c, g, lc in res:
        W[i, j] += w / n_races; C[i, j] += c / n_races
    return W, C
