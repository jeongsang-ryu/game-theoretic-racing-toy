"""Round-robin tournament.  Example: python scripts/run_tournament.py --races 16 --workers 4"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racing import tournament as T

ap = argparse.ArgumentParser()
ap.add_argument("--races", type=int, default=16, help="races per (defender, attacker) pair")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--out", type=str, default=str(T.DEFAULT_OUT))
a = ap.parse_args()
t0 = time.time()
res = T.run(a.races, a.workers, out=a.out)
W, C = T.summarize(res, a.races)
w = max(len(n) for n in T.NAMES)
print("defend success rate (rows = defender, cols = attacker)")
for i, n in enumerate(T.NAMES):
    print(f"{n:{w}s}", " ".join(f"{x:.2f}" for x in W[i]))
print("collisions per race")
for i, n in enumerate(T.NAMES):
    print(f"{n:{w}s}", " ".join(f"{x:4.1f}" for x in C[i]))
print(f"done in {time.time() - t0:.0f}s -> {a.out}")
