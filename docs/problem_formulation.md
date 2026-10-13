# How you define the race decides how you solve it

The same 1v1 race becomes a different mathematical problem depending on **how you treat the opponent**:
ignore it, predict it, assume it optimizes too, or learn against it.

## Common setup

- Cars $i \in \{1, 2\}$, state $x_i = (s_i, d_i, v_i)$: progress along the track, lateral offset, speed (Frenet frame).
- Input $u_i$ = (lateral rate, longitudinal acceleration).
- Dynamics $x_i^{k+1} = f(x_i^k, u_i^k)$, track bounds $|d_i| \le d_{\max}$, cornering limit $v_i \le \sqrt{a_{\text{lat}} R_{\text{eff}}(s_i, d_i)}$.
- **Interaction**: collisions ($|s_1 - s_2| < \ell$ and $|d_1 - d_2| < w$) and slipstream (the car right behind in the same lane gets a higher top speed).

Without interaction these are two independent optimal-control problems. Collisions and slipstream couple them.

## Four formulations

### 1. Single-agent optimal control (ignore the opponent)

$$\max_{u_i^{0:H}} \; s_i^H \quad \text{s.t. dynamics, track bounds, speed limits}$$

| Input | Output | Solver | Weakness |
|---|---|---|---|
| own state, track | own trajectory | MPC (receding horizon) | assumes an empty track: collisions, getting stuck |

### 2. Predict-then-plan

$$\max_{u_i} \; J_i(x, u_i, \hat u_j), \qquad \hat u_j = \text{prediction (e.g. keep lane and speed)}$$

| Input | Output | Solver | Weakness |
|---|---|---|---|
| own state, opponent state, prediction model | own trajectory | prediction + MPC (industry standard) | ignores that the opponent **reacts to you** |

### 3. Dynamic game

$$\text{for each } i: \; \max_{u_i} \; J_i(x, u_i, u_j) \quad \text{s.t. shared dynamics and collision constraints}$$

Solution concepts:

- **Nash equilibrium**: every player best-responds to the others; nobody gains by deviating alone.
- **Stackelberg**: a leader commits first, anticipating the follower's best response $BR_j(u_i)$. A natural model for the car in front defending.
- **Minimax (zero-sum)**: assume the worst-case opponent; for finite action sets this is a linear program (von Neumann; LP duality).
- **Generalized Nash**: players share constraints (collision avoidance).

| Input | Output | Solver | Weakness |
|---|---|---|---|
| both states, a model of the opponent's objective | an equilibrium trajectory pair (execute your own) | iterated best response, stacked KKT conditions (ALGAMES), LQ approximations (iLQGames), finite actions as a matrix game | compute cost, objective assumptions, multiple equilibria |

### 4. Markov game + reinforcement learning (self-play)

$$\max_\theta \; \mathbb{E}\Big[\sum_k \gamma^k r_i(x^k, a_i^k, a_j^k)\Big], \qquad a_i^k \sim \pi_\theta(\cdot \mid o_i^k)$$

| Input at run time | Output | Training | Weakness |
|---|---|---|---|
| observation $o_i$ (own state, relative opponent state, track preview) | action $a_i$ | PPO + self-play with a league of past snapshots | training cost, reward design, no constraint guarantees, sim-to-real |

## Side by side

| | Optimal control | Predict + plan | Game | Self-play RL |
|---|---|---|---|---|
| Opponent model | none | fixed prediction | **rational optimizer** | implicit, learned from play |
| When computed | online | online | online (heavy) | **offline training**, cheap online inference |
| Output | own trajectory | own trajectory | equilibrium pair | policy network $\pi_\theta$ |
| Guarantees | constraints | constraints | constraints, well-defined equilibrium | empirical |
| Interaction | none | one-way | **two-way** | two-way (implicit) |

## How the toy implements them

Every method chooses among the same **10 motion primitives** (2 s long; target lateral offset × speed mode),
so the continuous problem becomes a **finite-action game**. Each planner simulates all 10 × 10 joint primitive pairs
and builds a payoff matrix; the methods differ only in what they do with it.

| Method | Formulation | Rule on the 10 × 10 payoff matrix |
|---|---|---|
| Solo MPC | 1 | maximize own progress with the opponent removed |
| Predict+MPC | 2 | best response to "opponent keeps lane and speed" |
| Nash (IBR) | 3 | iterated best response to a pure Nash equilibrium (fallback: maximin) |
| Stackelberg | 3 | maximize own payoff given the opponent's best response to each action |
| Minimax (LP) | 3 | solve $\max_x \min_b x^\top U_{:,b}$ as an LP, sample from $x$ |
| Deep RL | 4 | PPO policy trained by self-play over the same 10 actions |
