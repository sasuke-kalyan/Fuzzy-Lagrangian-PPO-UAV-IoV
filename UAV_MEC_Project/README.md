# UAV-IoV-MEC LC-MAPO

Runnable research prototype for drone-aided secure task offloading in Internet of Vehicles using the LC-MAPO methodology.

## Paper Equations Implemented

- Drone action: `a_d(t) = (v_x,d(t), v_y,d(t))`
- Drone motion: `p_d(t+dt) = p_d(t) + a_d(t) * dt`
- Distance-based V2D assignment: `||p_v(t) - p_d(t)||_2 <= R_comm`, nearest reachable drone serves the task.
- Drone energy: `E_d(dt) = (P_idle + c_move * ||a_d(t)||_2^2) * dt`
- Battery update: `B_d(t+dt) = B_d(t) - E_d(dt)`
- Reward: `w_comp*N_comp - w_dist*d_min - w_en*E_step + C_time`
- Safety costs: inter-drone collision and low-battery indicators.
- LC-MAPO: TD3-style twin reward/cost critics with Lagrangian cost multipliers.

## Research Assumptions

The paper does not specify exact road coordinates, task arrival distributions, CPU queues, wireless channel rates, transmission delay, or neural-network layer sizes. This project therefore uses configurable assumptions:

- Synthetic road layouts for urban canyon, suburban crossroads, and emergency response.
- One-step task completion when a task is assigned to a reachable drone.
- Distance-only communication, no SINR or bandwidth model.
- Reward weights, task generation probability, and network hidden sizes are config parameters.
- Security is implemented as operational safety constraints: collision avoidance and low-battery avoidance.

## Setup

```bash
cd UAV_MEC_Project
python -m pip install -r requirements.txt
```

## Smoke Train LC-MAPO

```bash
python train_lc_mapo.py
```

The default config is intentionally small and should complete quickly. Results are written to `results/lc_mapo`, and the latest actor checkpoint is written to `checkpoints/lc_mapo_latest.pt`.

## Train MADDPG

```bash
python train_maddpg.py
```

## Evaluate Greedy Baseline

```bash
python evaluate.py --algorithm greedy --episodes 3
```

## Tests

```bash
python -m pytest tests
```

## Main Modules

- `envs/entities.py`: `Drone`, `Vehicle`, and `Task`.
- `envs/uav_iov_env.py`: multi-agent Gymnasium environment.
- `envs/energy.py`: paper energy and battery equations.
- `envs/reward.py`: paper reward shaping equation.
- `envs/costs.py`: collision and low-battery costs.
- `algorithms/lc_mapo`: LC-MAPO implementation.
- `algorithms/maddpg`: unconstrained MADDPG baseline.
- `algorithms/greedy`: nearest-task heuristic.
- `evaluation`: rollout and KPI aggregation.
- `visualization`: learning-curve plotting.
