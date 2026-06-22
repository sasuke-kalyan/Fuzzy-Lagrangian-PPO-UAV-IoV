from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values, window=250):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result


def get_xy(path):
    rows = load_log(path)
    x = [r["episode"] for r in rows]
    y = moving_average([r["reward"] for r in rows], window=250)
    return x, y


scenarios = [
    ("Urban Canyon Scenario", "urban_maddpg", "urban_lc_mapo"),
    ("Suburban Crossroads Scenario", "suburban_maddpg", "suburban_lc_mapo"),
    ("Emergency Response Scenario", "emergency_maddpg", "emergency_lc_mapo"),
]

out = Path("results/paper_style")
out.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 12))

for i, (title, maddpg_dir, lcmapo_dir) in enumerate(scenarios, start=1):
    plt.subplot(3, 1, i)

    x_m, y_m = get_xy(f"results/{maddpg_dir}/training_log.json")
    x_l, y_l = get_xy(f"results/{lcmapo_dir}/training_log.json")

    plt.plot(x_l, y_l, label="LC-MAPO", color="red", linewidth=2)
    plt.plot(x_m, y_m, label="MADDPG", color="blue", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"({chr(96+i)}) {title}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

plt.tight_layout()
plt.savefig(out / "episode_reward_all_scenarios.png", dpi=300)
plt.close()

print("Paper-style graph saved at results/paper_style/episode_reward_all_scenarios.png")