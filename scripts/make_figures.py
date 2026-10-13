"""Regenerate all figures and GIFs in figures/ from results/ and models/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racing import viz, agents, rl

F = viz.FIG
viz.heatmaps()
viz.learning_curve()
viz.timeseries([("Solo vs Solo", agents.Solo, agents.Solo, 3),
                ("Predict (def) vs Nash (att)", agents.Predictive, agents.Nash, 9),
                ("Stackelberg (def) vs Nash (att)", agents.Stackelberg, agents.Nash, 2),
                ("Minimax (def) vs Predict (att)", agents.Minimax, agents.Predictive, 1),
                ("RL (def) vs Nash (att)", rl.RLAgent, agents.Nash, 0)], F / "fig_cases.png")
viz.gif(agents.Predictive, agents.Nash, 9, F / "gif_overtake_nash.gif", "Predict+MPC (def) vs Nash IBR (att): overtake")
viz.gif(agents.Stackelberg, agents.Nash, 2, F / "gif_block_stackelberg.gif", "Stackelberg (def) vs Nash (att): aggressive block")
viz.gif(rl.RLAgent, agents.Nash, 0, F / "gif_rl_defense.gif", "Deep RL (def) vs Nash (att): learned inside cover")
viz.gif(agents.Solo, agents.Solo, 3, F / "gif_solo_crash.gif", "Solo MPC vs Solo MPC: ignoring the opponent")
print("figures written to", F)
