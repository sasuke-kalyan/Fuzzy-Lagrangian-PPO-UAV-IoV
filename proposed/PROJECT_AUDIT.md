# Project Audit: UAV-IoV RL Training System

**Date:** June 5, 2026  
**Repository:** /home/shubh-om/project_iov_uav-jun1/UAV_IOV

---

## 1. Current Architecture

### 1.1 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Scenario System | `scenarios.py` | Defines 3 paper-based scenarios with vehicle layouts, energy models, and task rates |
| RL Environment | `uav_iov_env.py` | Gymnasium environment for UAV selection with episode-based training |
| Training Pipeline | `train_scenario_rl.py` | Episode-based PPO training per scenario |
| Episode Callback | `episode_training_callback.py` | Logs per-episode rewards and training metrics |
| Reward Plotting | `reward_plotting.py` | Generates publication-style Episode vs Reward graphs |
| Training Tables | `training_tables.py` | Exports training results to XLSX and PNG tables |
| Evaluation Pipeline | `evaluation_pipeline.py` | Post-training evaluation with dataset metric graphs |
| Episode Horizon Graphs | `episode_horizon_graphs.py` | Generates metrics over 1000-step episode horizon |
| Results Management | `results_paths.py` | Centralized path management for all outputs |

### 1.2 Configuration Files

| File | Key Settings |
|------|--------------|
| `training_config.py` | `STEPS_PER_EPISODE = 1000`, `DEFAULT_TRAINING_EPISODES = 1000` |
| `episode_horizon_config.py` | `EPISODE_HORIZON_STEPS = 1000`, `TRAINING_EPISODES = 1000` |
| `network_config.py` | `NUM_UAVS = 8` |

### 1.3 Data Pipeline

| Component | File | Purpose |
|-----------|------|---------|
| Dataset Generation | `dataset_generation.py` | Generates synthetic UAV-IoV datasets |
| Data Loader | `data_loader.py` | Loads combined or per-scenario datasets |
| Communication Model | `communication_model.py` | Computes link metrics (delay, PDR, signal strength, energy) |

---

## 2. RL Training Pipeline

### 2.1 Episode-Based Training

**Implementation:** `train_scenario_rl.py`

- **Episodes:** 1000 per scenario (configurable via `--episodes` flag)
- **Steps per Episode:** 1000 (matches dataset timestamp horizon)
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** MLP with 128x128 hidden layers for actor and critic
- **Hyperparameters:**
  - Learning rate: 3e-4
  - Batch size: 64
  - N-steps: 512 (capped at 4x episode horizon)
  - N-epochs: 10
  - Gamma: 0.99
  - GAE lambda: 0.95
  - Clip range: 0.2
  - Entropy coefficient: 0.02

### 2.2 Episode Reward Logging

**Implementation:** `episode_training_callback.py`

- **Callback:** `EpisodeRewardCallback` integrates with Stable Baselines3 Monitor
- **Logged Metrics:**
  - Episode number
  - Total reward per episode
  - Average reward per step
  - Training time per episode
  - Number of steps per episode
- **Output:** JSON files in `results/training_logs/{scenario_id}_episode_rewards.json`

### 2.3 Scenario-Specific Training

**Implementation:** `train_scenario_rl.py::train_one_scenario()`

- Each scenario trained independently with separate:
  - PPO model instance
  - Environment instance
  - Training log
  - Model save file
- Seed offset ensures reproducibility across scenarios

---

## 3. Scenario System

### 3.1 Implemented Scenarios

| Scenario ID | Name | Vehicles | Layout | Focus |
|-------------|------|----------|--------|-------|
| `urban_canyon` | Urban Canyon | 20 | Corridor | Collision avoidance |
| `suburban_crossroads` | Suburban Crossroads | 15 | Sparse grid | Energy efficiency |
| `emergency_response` | Emergency Response | 5 → 15 | Emergency dynamic | Dynamic adaptation |

### 3.2 Scenario Configuration

**File:** `scenarios.py`

Each scenario defines:
- Vehicle count and layout type
- Speed ranges and movement patterns
- UAV movement patterns
- Energy drain multipliers
- Delay multipliers
- PDR offsets
- Task rates (high/medium/low)
- Emergency vehicle injection (for emergency_response)
- Operational area: 2000m x 2000m

### 3.3 Emergency Scenario Dynamics

- **Initial state:** 5 vehicles, low task rate
- **Emergency trigger:** At step 333 (120s scaled from 360s total)
- **Emergency response:** +10 vehicles injected, burst task load
- **Purpose:** Tests non-stationary adaptation

---

## 4. Evaluation Pipeline

### 4.1 Post-Training Evaluation

**Implementation:** `evaluation_pipeline.py::run_post_training_evaluation()`

The pipeline executes three stages:

1. **Episode Horizon Graphs** (`episode_horizon_graphs.py`)
   - Delay over 1000-step horizon
   - PDR over 1000-step horizon
   - Throughput over 1000-step horizon
   - Energy over 1000-step horizon

2. **Dataset Metric Graphs** (`evaluation_pipeline.py::generate_dataset_metric_graphs()`)
   - Communication delay
   - Packet Delivery Ratio (PDR)
   - Throughput proxy
   - UAV energy
   - UAV utilization (bar chart)
   - Network lifetime proxy

3. **Legacy Analysis Scripts**
   - `pdr_analysis.py`
   - `delay_analysis.py`
   - `reliability_score.py`
   - `final_performance_comparison.py`
   - `scenario_analysis.py`
   - `energy_optimization.py`

### 4.2 Graph Generation Pipeline

**Output Directories:**
- `results/episode_horizon/` - Episode horizon metrics
- `results/reward_graphs/` - Episode vs Reward graphs
- `results/delay_graphs/` - Communication delay graphs
- `results/pdr_graphs/` - Packet delivery ratio graphs
- `results/throughput_graphs/` - Throughput graphs
- `results/energy_graphs/` - UAV energy graphs
- `results/utilization_graphs/` - UAV utilization graphs
- `results/lifetime_graphs/` - Network lifetime graphs
- `results/training_tables/` - Training result tables

---

## 5. Graph Generation Pipeline

### 5.1 Reward Graphs

**Implementation:** `reward_plotting.py`

- **Type:** Episode vs Episode Reward (single smooth line)
- **Style:** Publication-quality with IEEE formatting
- **Smoothing:** Moving average with adaptive window (40 for 1000 episodes)
- **Output:** One graph per scenario in `results/reward_graphs/`
- **Features:**
  - Fixed x-axis to 1000 episodes
  - Red line (#C44E52) for proposed method
  - Scenario caption below plot
  - Grid lines and legend

### 5.2 Episode Horizon Graphs

**Implementation:** `episode_horizon_graphs.py`

- **Type:** Metrics over 1000-step episode horizon
- **Metrics:** Delay, PDR, Throughput, Energy
- **Style:** Publication-quality with smoothing (window=25)
- **Output:** 4 graphs per scenario in `results/episode_horizon/`
- **X-axis:** Step within episode horizon (0-1000)
- **Color:** Red line (#C44E52) for proposed method

### 5.3 Dataset Metric Graphs

**Implementation:** `evaluation_pipeline.py::generate_dataset_metric_graphs()`

- **Type:** Time-series metrics from dataset
- **Metrics:** Delay, PDR, Throughput, Energy, Utilization, Lifetime
- **Style:** Publication-quality with smoothing
- **Output:** 6 graphs per scenario across respective directories
- **X-axis:** Timestamp / Step within episode horizon

---

## 6. Results/Output Structure

### 6.1 Directory Structure

```
UAV_IOV/
├── results/
│   ├── delay_graphs/           [EMPTY - needs evaluation pipeline]
│   ├── energy_graphs/          [EMPTY - needs evaluation pipeline]
│   ├── episode_horizon/        [POPULATED - 12 graphs]
│   ├── lifetime_graphs/        [EMPTY - needs evaluation pipeline]
│   ├── pdr_graphs/             [EMPTY - needs evaluation pipeline]
│   ├── reward_graphs/          [POPULATED - 3 graphs]
│   ├── throughput_graphs/      [EMPTY - needs evaluation pipeline]
│   ├── training_logs/          [POPULATED - 3 JSON files]
│   ├── training_tables/        [POPULATED - 6 files (3 xlsx + 3 png)]
│   └── utilization_graphs/     [EMPTY - needs evaluation pipeline]
├── models/
│   ├── ppo_urban_canyon.zip    [EXISTS]
│   ├── ppo_suburban_crossroads.zip [EXISTS]
│   └── ppo_emergency_response.zip [EXISTS]
└── graph_outputs/
    ├── [Legacy graphs from earlier analysis]
    └── scenarios/              [EMPTY]
```

### 6.2 Training Logs

**Location:** `results/training_logs/`

| File | Episodes | Steps/Episode |
|------|----------|---------------|
| `urban_canyon_episode_rewards.json` | 1000 | 1000 |
| `suburban_crossroads_episode_rewards.json` | 1000 | 1000 |
| `emergency_response_episode_rewards.json` | 1000 | 1000 |

**Format:** JSON with episode records containing:
- Episode number
- Total reward
- Average reward
- Scenario name
- Training time

### 6.3 Reward Graphs

**Location:** `results/reward_graphs/`

| File | Scenario |
|------|----------|
| `urban_episode_reward.png` | Urban Canyon |
| `suburban_episode_reward.png` | Suburban Crossroads |
| `emergency_episode_reward.png` | Emergency Response |

**Status:** All 3 graphs exist with publication-quality formatting

### 6.4 Episode Horizon Graphs

**Location:** `results/episode_horizon/`

| File | Scenario | Metric |
|------|----------|--------|
| `urban_delay_horizon.png` | Urban Canyon | Delay |
| `urban_pdr_horizon.png` | Urban Canyon | PDR |
| `urban_throughput_horizon.png` | Urban Canyon | Throughput |
| `urban_energy_horizon.png` | Urban Canyon | Energy |
| `suburban_delay_horizon.png` | Suburban Crossroads | Delay |
| `suburban_pdr_horizon.png` | Suburban Crossroads | PDR |
| `suburban_throughput_horizon.png` | Suburban Crossroads | Throughput |
| `suburban_energy_horizon.png` | Suburban Crossroads | Energy |
| `emergency_delay_horizon.png` | Emergency Response | Delay |
| `emergency_pdr_horizon.png` | Emergency Response | PDR |
| `emergency_throughput_horizon.png` | Emergency Response | Throughput |
| `emergency_energy_horizon.png` | Emergency Response | Energy |

**Status:** All 12 graphs exist (4 metrics × 3 scenarios)

### 6.5 Training Tables

**Location:** `results/training_tables/`

| File | Scenario | Type |
|------|----------|------|
| `urban_training_results.xlsx` | Urban Canyon | Excel |
| `urban_training_table.png` | Urban Canyon | PNG preview |
| `suburban_training_results.xlsx` | Suburban Crossroads | Excel |
| `suburban_training_table.png` | Suburban Crossroads | PNG preview |
| `emergency_training_results.xlsx` | Emergency Response | Excel |
| `emergency_training_table.png` | Emergency Response | PNG preview |

**Status:** All 6 files exist (3 scenarios × 2 formats)

### 6.6 PPO Models

**Location:** `models/`

| File | Scenario |
|------|----------|
| `ppo_urban_canyon.zip` | Urban Canyon |
| `ppo_suburban_crossroads.zip` | Suburban Crossroads |
| `ppo_emergency_response.zip` | Emergency Response |

**Status:** All 3 models exist (550KB each)

### 6.7 Missing Evaluation Graphs

**Empty Directories (need evaluation pipeline run):**

- `results/delay_graphs/` - 0 files
- `results/pdr_graphs/` - 0 files
- `results/throughput_graphs/` - 0 files
- `results/energy_graphs/` - 0 files
- `results/utilization_graphs/` - 0 files
- `results/lifetime_graphs/` - 0 files

**Expected Output:** 18 graphs (6 metrics × 3 scenarios)

---

## 7. Verification Against Intended Goals

### 7.1 Target Specifications

| Requirement | Status | Details |
|-------------|--------|---------|
| 3 paper scenarios | ✅ COMPLETE | urban_canyon, suburban_crossroads, emergency_response |
| 1000 training episodes | ✅ COMPLETE | All scenarios have 1000 episodes logged |
| 1000 steps per episode | ✅ COMPLETE | STEPS_PER_EPISODE = 1000 |
| Episode vs Reward graph per scenario | ✅ COMPLETE | 3 graphs in results/reward_graphs/ |
| Episode horizon graphs | ✅ COMPLETE | 12 graphs in results/episode_horizon/ |
| Publication-quality graphs | ✅ COMPLETE | IEEE-style formatting, 300 DPI |
| Training result tables | ✅ COMPLETE | 3 XLSX + 3 PNG tables |
| PPO models | ✅ COMPLETE | 3 models saved |
| Episode reward logging | ✅ COMPLETE | JSON logs with per-episode metrics |
| Scenario-specific PPO training | ✅ COMPLETE | Independent training per scenario |

### 7.2 Missing Components

| Requirement | Status | Details |
|-------------|--------|---------|
| Delay graphs | ❌ MISSING | results/delay_graphs/ is empty |
| PDR graphs | ❌ MISSING | results/pdr_graphs/ is empty |
| Throughput graphs | ❌ MISSING | results/throughput_graphs/ is empty |
| Energy graphs | ❌ MISSING | results/energy_graphs/ is empty |
| Utilization graphs | ❌ MISSING | results/utilization_graphs/ is empty |
| Lifetime graphs | ❌ MISSING | results/lifetime_graphs/ is empty |

**Root Cause:** The evaluation pipeline exists (`evaluation_pipeline.py`) but the `generate_dataset_metric_graphs()` function has not been executed to populate these directories.

---

## 8. Key Findings

### 8.1 Strengths

1. **Complete RL Training Pipeline:** Episode-based training with 1000 episodes × 1000 steps is fully implemented and executed
2. **Comprehensive Logging:** Per-episode rewards, training times, and metrics are logged in JSON format
3. **Publication-Quality Visualizations:** Reward graphs and episode horizon graphs use IEEE-style formatting
4. **Scenario System:** All 3 paper scenarios are properly configured with distinct characteristics
5. **Model Persistence:** All 3 PPO models are saved and can be loaded for evaluation
6. **Training Tables:** Excel and PNG table exports provide human-readable training summaries

### 8.2 Gaps

1. **Incomplete Evaluation:** Dataset metric graphs (delay, PDR, throughput, energy, utilization, lifetime) are missing
2. **Pipeline Not Executed:** The evaluation pipeline exists but hasn't been run to completion
3. **Legacy Graph Outputs:** The `graph_outputs/` directory contains legacy graphs but the structured `results/` subdirectories are incomplete

### 8.3 Architecture Quality

- **Modular Design:** Clear separation between scenarios, training, evaluation, and visualization
- **Configuration Management:** Centralized path management and training configuration
- **Reproducibility:** Seed management and deterministic scenario configurations
- **Extensibility:** Easy to add new scenarios or modify training parameters

---

## 9. Recommendations

### 9.1 Immediate Actions

1. **Run Evaluation Pipeline:** Execute `python evaluation_pipeline.py` to generate missing dataset metric graphs
2. **Verify Graph Outputs:** Ensure all 18 expected graphs are generated (6 metrics × 3 scenarios)
3. **Update Documentation:** Create PROJECT_STATUS.md to track completion status

### 9.2 Future Enhancements

1. **Automated Validation:** Add verification script to check for all expected outputs
2. **Comparison Baselines:** Implement traditional/baseline method comparisons for graphs
3. **Statistical Analysis:** Add confidence intervals and statistical significance testing
4. **Hyperparameter Tuning:** Document hyperparameter selection rationale
5. **Ablation Studies:** Add ablation study components for paper submission

---

## 10. Conclusion

The repository contains a **complete and functional RL training pipeline** for UAV-IoV optimization with all core requirements met:

- ✅ 3 paper scenarios implemented
- ✅ 1000-episode training completed
- ✅ 1000-step episode horizon configured
- ✅ Episode reward logging implemented
- ✅ Scenario-specific PPO training executed
- ✅ Reward graphs generated
- ✅ Training tables exported
- ✅ PPO models saved
- ✅ Episode horizon graphs generated

The **only missing component** is the execution of the evaluation pipeline to generate the dataset metric graphs (delay, PDR, throughput, energy, utilization, lifetime). The pipeline code exists and is ready to run.

**Overall Assessment:** 85% complete - training phase finished, evaluation phase partially complete.
