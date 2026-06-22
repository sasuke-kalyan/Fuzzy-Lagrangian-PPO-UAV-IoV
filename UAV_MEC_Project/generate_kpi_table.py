from pathlib import Path
import json
import pandas as pd


def load_log(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average(log, key):
    values = [float(row.get(key, 0.0)) for row in log]
    return sum(values) / max(len(values), 1)


BASE_DIR = Path("results")

experiments = [
    # Urban Canyon
    ("Urban Canyon", "MADDPG",
     BASE_DIR / "qos_urban_maddpg" / "training_log.json"),

    ("Urban Canyon", "LC-MAPO",
     BASE_DIR / "qos_urban_lc_mapo" / "training_log.json"),

    ("Urban Canyon", "Greedy-PPO",
     BASE_DIR / "qos_test" / "training_log.json"),

    # Suburban Crossroads
    ("Suburban Crossroads", "MADDPG",
     BASE_DIR / "qos_suburban_maddpg" / "training_log.json"),

    ("Suburban Crossroads", "LC-MAPO",
     BASE_DIR / "qos_suburban_lc_mapo" / "training_log.json"),

    ("Suburban Crossroads", "Greedy-PPO",
     BASE_DIR / "qos_suburban_greedy_ppo" / "training_log.json"),

    # Emergency Response
    ("Emergency Response", "MADDPG",
     BASE_DIR / "qos_emergency_maddpg" / "training_log.json"),

    ("Emergency Response", "LC-MAPO",
     BASE_DIR / "qos_emergency_lc_mapo" / "training_log.json"),

    ("Emergency Response", "Greedy-PPO",
     BASE_DIR / "qos_emergency_greedy_ppo" / "training_log.json"),
]

rows = []

for scenario, algorithm, file_path in experiments:
    log = load_log(file_path)

    avg_reward = average(log, "reward")
    avg_tasks = average(log, "completed_tasks")
    avg_energy = average(log, "total_energy_j")
    avg_collision = average(log, "collision_events")
    avg_battery = average(log, "low_battery_events")

    avg_delay = average(log, "avg_delay_ms")
    avg_pdr = average(log, "avg_pdr_pct")
    avg_signal = average(log, "avg_signal")
    avg_energy_pct = average(log, "avg_energy_pct")
    avg_qos_penalty = average(log, "qos_penalty")

    energy_per_task = avg_energy / max(avg_tasks, 1.0)

    rows.append({
        "Scenario": scenario,
        "Algorithm": algorithm,
        "Average Reward": round(avg_reward, 3),
        "Completed Tasks": round(avg_tasks, 3),
        "Energy (J)": round(avg_energy, 3),
        "Energy per Task": round(energy_per_task, 3),
        "Collision Events": round(avg_collision, 3),
        "Low Battery Events": round(avg_battery, 3),
        "Average Delay (ms)": round(avg_delay, 3),
        "Average PDR (%)": round(avg_pdr, 3),
        "Average Signal": round(avg_signal, 3),
        "Average Energy (%)": round(avg_energy_pct, 3),
        "QoS Penalty": round(avg_qos_penalty, 3)
    })

table = pd.DataFrame(rows)

output_dir = BASE_DIR / "comparison"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "qos_all_scenarios_kpi_table.csv"

table.to_csv(csv_path, index=False)

print("\n========== FINAL QoS KPI TABLE ==========\n")
print(table)
print("\nSaved successfully!")
print(f"Location: {csv_path}")