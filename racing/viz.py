import json, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from . import sim, race, agents, rl

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.json"
FIG = ROOT / "figures"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C_DEF = "#2563eb"   # defender (car0) blue
C_ATT = "#ea580c"   # attacker (car1) orange
SHORT = {"Solo MPC": "Solo", "Predict+MPC": "Predict", "Nash (IBR)": "Nash",
         "Stackelberg": "Stackel.", "Minimax (LP)": "Minimax", "Deep RL (self-play)": "RL"}


def draw_track(ax):
    s = np.linspace(0, sim.L, 800)
    for dd, ls, lw in [(5, "-", 1.5), (-5, "-", 1.5), (0, "--", 0.6)]:
        x, y = sim.xy(s, np.full_like(s, dd))
        ax.plot(x, y, ls, color="#9ca3af", lw=lw)
    ax.set_aspect("equal"); ax.axis("off")


def heatmaps():
    R = json.load(open(RESULTS))
    names = [SHORT[n] for n in R["names"]]; n = len(names); N = R["N"]
    W = np.zeros((n, n)); C = np.zeros((n, n))
    for i, j, s, w, c, g, lc in R["results"]:
        W[i, j] += w / N; C[i, j] += c / N
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, M, title, cmap, fmt in [(axs[0], W, "Defend success rate", "Blues", "{:.2f}"),
                                    (axs[1], C, "Collisions per race", "Reds", "{:.1f}")]:
        im = ax.imshow(M, cmap=cmap)
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30)
        ax.set_yticks(range(n)); ax.set_yticklabels(names)
        ax.set_xlabel("Attacker (faster car, starts behind)"); ax.set_ylabel("Defender (starts ahead)")
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > M.max() * 0.6 else "black", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "fig_heatmaps.png", dpi=150); plt.close()

    # summary: score = avg(defend rate as defender, attack success as attacker)
    defend = W.mean(1); attack = 1 - W.mean(0)
    coll = (C.mean(1) + C.mean(0)) / 2
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(n); w = 0.38
    axs[0].bar(x - w / 2, defend, w, color=C_DEF, label="as defender: hold position")
    axs[0].bar(x + w / 2, attack, w, color=C_ATT, label="as attacker: overtake")
    axs[0].set_xticks(x); axs[0].set_xticklabels(names); axs[0].set_ylim(0, 1.25)
    axs[0].set_title("Success rate (avg over all opponents)"); axs[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    axs[1].bar(x, coll, color="#dc2626"); axs[1].set_xticks(x); axs[1].set_xticklabels(names)
    axs[1].set_title("Collisions per race (avg over both roles)")
    plt.tight_layout(); plt.savefig(FIG / "fig_summary.png", dpi=150); plt.close()
    return names, W, C, defend, attack, coll


def learning_curve():
    log = pickle.load(open(rl.DEFAULT_MODEL, "rb"))["log"]
    it = np.array([l[0] for l in log]); ret = np.array([l[1] for l in log])
    k = 15; sm = np.convolve(ret, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(6, 3.5))
    plt.plot(it, ret, color="#d1d5db", lw=0.8, label="per iteration")
    plt.plot(it[k - 1:], sm, color="#7c3aed", lw=2, label="moving avg")
    plt.xlabel("PPO iteration"); plt.ylabel("episode return (learner)")
    plt.title("Self-play training"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(FIG / "fig_rl_curve.png", dpi=150); plt.close()


def race_traj(defA, attA, seed):
    rng = np.random.default_rng(seed)
    S = race.init_state(rng)
    return race.run_race([defA(), attA()], S, np.random.default_rng(seed + 1000), log=True)


def snapshot_panel(cases, fname):
    """cases: list of (title, defender cls, attacker cls, seed). Plot trajectories over time."""
    fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.6))
    for ax, (title, D, A, seed) in zip(np.atleast_1d(axs), cases):
        r = race_traj(D, A, seed)
        draw_track(ax)
        tr = r["traj"]
        s = np.array([t[0] for t in tr]); d = np.array([t[1] for t in tr])
        for c, col in [(0, C_DEF), (1, C_ATT)]:
            x, y = sim.xy(s[:, c], d[:, c])
            ax.plot(x, y, color=col, lw=1.2, alpha=0.8)
        coll_idx = [k for k, t in enumerate(tr) if t[4]]
        for k in coll_idx:
            x, y = sim.xy(s[k:k + 1, 0], d[k:k + 1, 0]); ax.plot(x, y, "x", color="#dc2626", ms=9, mew=2)
        res = "defender holds" if r["winner"] == 0 else "attacker passes"
        ax.set_title(f"{title}\n{res}, collisions={r['collisions']}", fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def gif(defA, attA, seed, fname, title, T=None, every=3):
    r = race_traj(defA, attA, seed)
    tr = r["traj"] if T is None else r["traj"][:int(T / sim.DT)]
    s = np.array([t[0] for t in tr]); d = np.array([t[1] for t in tr])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    draw_track(ax)
    dots = [ax.plot([], [], "o", color=c, ms=9)[0] for c in (C_DEF, C_ATT)]
    tails = [ax.plot([], [], "-", color=c, lw=2, alpha=0.5)[0] for c in (C_DEF, C_ATT)]
    txt = ax.text(0, 0, "", ha="center", va="center", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.text(-40, -40, "● defender", color=C_DEF, fontsize=9); ax.text(5, -40, "● attacker (faster car)", color=C_ATT, fontsize=9); ax.set_ylim(-45, 35)

    def upd(f):
        k = f * every
        for c in range(2):
            x, y = sim.xy(s[k:k + 1, c], d[k:k + 1, c]); dots[c].set_data(x, y)
            k0 = max(0, k - 15)
            x, y = sim.xy(s[k0:k + 1, c], d[k0:k + 1, c]); tails[c].set_data(x, y)
        gap = s[k, 0] - s[k, 1]
        txt.set_text(f"t={k * sim.DT:4.1f}s   gap={gap:+5.1f} m\n" + ("defender ahead" if gap > 0 else "attacker ahead"))
        return dots + tails + [txt]

    anim = animation.FuncAnimation(fig, upd, frames=len(tr) // every, blit=True)
    anim.save(str(fname), writer=animation.PillowWriter(fps=12)); plt.close()
    return r


def timeseries(cases, fname):
    """per case: top = gap (defender ahead > 0), bottom = lateral position of both cars."""
    n = len(cases)
    fig, axs = plt.subplots(2, n, figsize=(4.3 * n, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    for c, (title, D, A, seed) in enumerate(cases):
        r = race_traj(D, A, seed); tr = r["traj"]
        t = np.arange(len(tr)) * sim.DT
        s = np.array([x[0] for x in tr]); d = np.array([x[1] for x in tr])
        gap = s[:, 0] - s[:, 1]
        ax = axs[0, c]
        ax.axhline(0, color="#9ca3af", lw=0.8)
        ax.fill_between(t, gap, 0, where=gap > 0, color=C_DEF, alpha=0.25)
        ax.fill_between(t, gap, 0, where=gap <= 0, color=C_ATT, alpha=0.35)
        ax.plot(t, gap, color="#111827", lw=1.2)
        for k, x in enumerate(tr):
            if x[4]: ax.axvline(t[k], color="#dc2626", lw=0.8, alpha=0.6)
        res = "defender holds" if r["winner"] == 0 else "attacker passes"
        ax.set_title(f"{title}\n{res} · collisions {r['collisions']}", fontsize=10)
        if c == 0: ax.set_ylabel("gap [m]\n(+ defender ahead)")
        ax2 = axs[1, c]
        ax2.plot(t, d[:, 0], color=C_DEF, lw=1.4, label="defender")
        ax2.plot(t, d[:, 1], color=C_ATT, lw=1.4, label="attacker")
        kk = sim.kappa(s[:, 0]) > 0
        ax2.fill_between(t, -5, 5, where=kk, color="#e5e7eb", alpha=0.7, lw=0)
        ax2.set_ylim(-5, 5); ax2.set_xlabel("time [s]")
        if c == 0:
            ax2.set_ylabel("lateral pos d [m]\n(+ inside)"); ax2.legend(frameon=False, fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()
