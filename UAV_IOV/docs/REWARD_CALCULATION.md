# How Episode Reward Is Calculated (Proposed UAV-IoV / PPO)

## Episode reward (what the graph plots)

Each **training episode** has **50 steps** (same as dataset timestamps).

```
Episode Reward = sum of step rewards for steps 1 … 50
```

The graph shows this total per episode.
**Average Reward** in Excel = `Episode Reward / 50`.

---

## Step reward (each UAV selection)

At every step the agent picks one UAV. The environment computes link metrics, then uses
the same baseline-style reward structure:

```
step_reward = task_reward - distance_penalty - energy_penalty + time_penalty
            + QoS_reward
```

### Step 1 — Link metrics (`compute_link_metrics`)

| Quantity | Meaning |
|----------|---------|
| **Distance** | 3D distance vehicle ↔ UAV |
| **Signal** | 0.05 (far) to ~1.0 (close, within 500 m) |
| **Delay** | ms, from distance |
| **PDR** | % packet delivery estimate from signal and delay |

### Step 2 — Baseline task/reward terms

These match the baseline project weights:

```
task_reward      = 15.0 × completed_count
distance_penalty = 0.002 × selected_UAV_distance
energy_penalty   = 0.001 × selected_UAV_energy_used
time_penalty     = -0.05
```

Because `UAV_IOV` makes one UAV-selection/offloading decision per step, the
environment uses `completed_count = 1` for that selected task.

### Step 3 — Shared QoS score (`qos_score`)

This uses the same QoS reward equation as the baseline project:

```
energy_pct = min(100, UAV_energy × 2)

score = 120 × signal
      + 1.5 × PDR
      − 1.2 × delay
      + 0.8 × energy_pct
```

### Step 4 — Soft QoS penalties (`constraint_penalty`)

Small penalties only if QoS limits are broken (loosened thresholds):

| Constraint | Threshold |
|------------|-----------|
| Delay | ≤ 100 ms |
| PDR | ≥ 48% |
| Energy | ≥ 8% |
| Signal | ≥ 0.05 |

```
penalty = 0.25×delay_violation + 0.15×pdr_violation
        + 0.15×energy_violation + 0.08×signal_violation
```

### Step 5 — Final step reward (`baseline_compatible_reward`)

```
QoS_reward = score − penalty

step_reward = task_reward
            − distance_penalty
            − energy_penalty
            + time_penalty
            + QoS_reward
```

Good UAV choices (strong signal, higher PDR, lower delay, enough energy) produce
a larger reward. Poor choices can now produce low or negative reward; there is no
forced positive floor.

---

## Why your values differ from the paper figure (35k–60k)

The reference paper (LC-MAPO / MADDPG) uses a **multi-drone, multi-vehicle** simulator with a **different reward definition** and often **more agents and longer horizons**.  

This project uses **single-vehicle UAV selection** and sums **50 step rewards** per episode.

To get a curve shape like the paper (smooth rise over ~1500 episodes), train with:

```bash
python train_ppo.py --episodes 1500
```

---

## Code locations

| File | Role |
|------|------|
| `communication_model.py` | `baseline_compatible_reward`, `shaped_reward`, `qos_score` |
| `constraints.py` | QoS thresholds |
| `uav_iov_env.py` | Calls `shaped_reward` in `step()` |
| `episode_training_callback.py` | Sums episode return from Monitor |
| `reward_plotting.py` | One-line smoothed Episode vs Reward plot |
