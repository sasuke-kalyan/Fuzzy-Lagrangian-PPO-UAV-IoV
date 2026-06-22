from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=100):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i + 1]))
    return result


def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


BASE_DIR = Path("results")

maddpg_log = load_log(BASE_DIR / "maddpg" / "training_log.json")
lcmapo_log = load_log(BASE_DIR / "lc_mapo" / "training_log.json")

episodes = [x["episode"] for x in maddpg_log]

reward_maddpg = [x["reward"] for x in maddpg_log]
reward_lcmapo = [x["reward"] for x in lcmapo_log]

tasks_maddpg = [x["completed_tasks"] for x in maddpg_log]
tasks_lcmapo = [x["completed_tasks"] for x in lcmapo_log]

energy_maddpg = [x["total_energy_j"] for x in maddpg_log]
energy_lcmapo = [x["total_energy_j"] for x in lcmapo_log]

reward_maddpg = moving_average(reward_maddpg, 100)
reward_lcmapo = moving_average(reward_lcmapo, 100)

tasks_maddpg = moving_average(tasks_maddpg, 100)
tasks_lcmapo = moving_average(tasks_lcmapo, 100)

energy_maddpg = moving_average(energy_maddpg, 100)
energy_lcmapo = moving_average(energy_lcmapo, 100)

output_dir = BASE_DIR / "comparison"
output_dir.mkdir(exist_ok=True)

# Reward Comparison
plt.figure(figsize=(10, 6))
plt.plot(episodes, reward_maddpg, label="MADDPG")
plt.plot(episodes, reward_lcmapo, label="LC-MAPO")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "reward_comparison.png", dpi=300)
plt.close()

# Task Completion Comparison
plt.figure(figsize=(10, 6))
plt.plot(episodes, tasks_maddpg, label="MADDPG")
plt.plot(episodes, tasks_lcmapo, label="LC-MAPO")
plt.xlabel("Episode")
plt.ylabel("Completed Tasks")
plt.title("Task Completion Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "task_comparison.png", dpi=300)
plt.close()

# Energy Comparison
plt.figure(figsize=(10, 6))
plt.plot(episodes, energy_maddpg, label="MADDPG")
plt.plot(episodes, energy_lcmapo, label="LC-MAPO")
plt.xlabel("Episode")
plt.ylabel("Energy (J)")
plt.title("Energy Consumption Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "energy_comparison.png", dpi=300)
plt.close()

print("Comparison graphs generated successfully!")