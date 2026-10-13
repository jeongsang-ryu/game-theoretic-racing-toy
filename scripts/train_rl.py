"""Train the self-play PPO policy.  Example: python scripts/train_rl.py --iters 1000"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racing import rl

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--batch", type=int, default=128, help="parallel environments per iteration")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=str, default=str(rl.DEFAULT_MODEL_OUT))
a = ap.parse_args()
rl.train(iters=a.iters, B=a.batch, seed=a.seed, out=a.out)
