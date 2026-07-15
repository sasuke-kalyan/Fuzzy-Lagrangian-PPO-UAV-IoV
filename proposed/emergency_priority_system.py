import numpy as np
import pandas as pd

from data_loader import load_dataset
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# 1. Load dataset (unchanged)
# ---------------------------------------------------------------------------
df = load_dataset()

print("\n======================================")
print(" Fuzzy Emergency Vehicle Priority System ")
print("======================================\n")

# ---------------------------------------------------------------------------
# 2. Random vehicle types (same logic as your original script)
# ---------------------------------------------------------------------------
vehicle_types = []
for _ in range(len(df)):
    rand = np.random.randint(1, 10)
    if rand == 1:
        vehicle_types.append("Ambulance")
    elif rand == 2:
        vehicle_types.append("Police")
    else:
        vehicle_types.append("Normal")

df["Vehicle_Type"] = vehicle_types

# Map vehicle role to crisp urgency in [0, 1] for fuzzy input
URGENCY_MAP = {
    "Normal": 0.0,
    "Police": 0.65,
    "Ambulance": 1.0,
}
df["Urgency"] = df["Vehicle_Type"].map(URGENCY_MAP)

# ---------------------------------------------------------------------------
# 3. Fuzzy sets (membership functions tuned to your dataset ranges)
#    Signal ~ 0.05–0.74, Delay ~ 14–100, PDR ~ 50–67
# ---------------------------------------------------------------------------

# --- Inputs ---
signal = ctrl.Antecedent(np.arange(0, 1.001, 0.01), "signal")
delay = ctrl.Antecedent(np.arange(0, 110.1, 1.0), "delay")
pdr = ctrl.Antecedent(np.arange(0, 101, 1.0), "pdr")
urgency = ctrl.Antecedent(np.arange(0, 1.001, 0.01), "urgency")

# Signal strength [0, 1]
signal["weak"] = fuzz.trimf(signal.universe, [0.00, 0.00, 0.30])
signal["medium"] = fuzz.trimf(signal.universe, [0.18, 0.40, 0.62])
signal["strong"] = fuzz.trimf(signal.universe, [0.50, 0.72, 1.00])

# Delay (ms) — lower is better
delay["low"] = fuzz.trimf(delay.universe, [0, 0, 35])
delay["medium"] = fuzz.trimf(delay.universe, [25, 50, 75])
delay["high"] = fuzz.trimf(delay.universe, [60, 100, 110])

# PDR (%) — higher is better
pdr["poor"] = fuzz.trimf(pdr.universe, [0, 0, 52])
pdr["acceptable"] = fuzz.trimf(pdr.universe, [48, 55, 62])
pdr["good"] = fuzz.trimf(pdr.universe, [58, 68, 100])

# Emergency urgency [0, 1] from vehicle type
urgency["low"] = fuzz.trimf(urgency.universe, [0.0, 0.0, 0.45])
urgency["medium"] = fuzz.trimf(urgency.universe, [0.30, 0.60, 0.85])
urgency["high"] = fuzz.trimf(urgency.universe, [0.70, 1.0, 1.0])

# --- Output: communication priority [0, 100] ---
priority = ctrl.Consequent(np.arange(0, 101, 1), "priority")

priority["very_low"] = fuzz.trimf(priority.universe, [0, 0, 20])
priority["low"] = fuzz.trimf(priority.universe, [10, 25, 40])
priority["medium"] = fuzz.trimf(priority.universe, [30, 50, 70])
priority["high"] = fuzz.trimf(priority.universe, [55, 75, 90])
priority["very_high"] = fuzz.trimf(priority.universe, [80, 100, 100])

# ---------------------------------------------------------------------------
# 4. Fuzzy rule base
# ---------------------------------------------------------------------------
rules = [
    # Emergency vehicles dominate priority
    ctrl.Rule(urgency["high"], priority["very_high"]),
    ctrl.Rule(urgency["medium"], priority["high"]),

    # Good link quality -> high priority
    ctrl.Rule(
        signal["strong"] & delay["low"] & pdr["good"],
        priority["very_high"],
    ),
    ctrl.Rule(signal["strong"] & delay["low"], priority["high"]),
    ctrl.Rule(signal["medium"] & delay["low"] & pdr["acceptable"], priority["high"]),

    # Average conditions
    ctrl.Rule(signal["medium"] & delay["medium"], priority["medium"]),
    ctrl.Rule(urgency["low"] & signal["strong"] & delay["low"], priority["medium"]),

    # Poor link -> low priority (unless emergency rules fired above)
    ctrl.Rule(signal["weak"] | delay["high"], priority["low"]),
    ctrl.Rule(pdr["poor"], priority["low"]),
    ctrl.Rule(urgency["low"] & signal["weak"] & delay["high"], priority["very_low"]),
]

priority_ctrl = ctrl.ControlSystem(rules)

# ---------------------------------------------------------------------------
# 5. Evaluate fuzzy priority for each row
# ---------------------------------------------------------------------------
priority_scores = []
fuzzy_labels = []

for _, row in df.iterrows():
    sim = ctrl.ControlSystemSimulation(priority_ctrl, clip_to_bounds=True)

    sim.input["signal"] = float(row["Signal_Strength"])
    sim.input["delay"] = float(row["Delay"])
    sim.input["pdr"] = float(row["PDR"])
    sim.input["urgency"] = float(row["Urgency"])

    try:
        sim.compute()
        score = float(sim.output["priority"])
    except ValueError:
        # Rare edge case: no rule fires strongly -> neutral default
        score = 50.0

    priority_scores.append(round(score, 2))

    # Optional linguistic label (for reporting)
    if score >= 80:
        fuzzy_labels.append("Very High")
    elif score >= 65:
        fuzzy_labels.append("High")
    elif score >= 45:
        fuzzy_labels.append("Medium")
    elif score >= 25:
        fuzzy_labels.append("Low")
    else:
        fuzzy_labels.append("Very Low")

df["Priority_Score"] = priority_scores
df["Priority_Label"] = fuzzy_labels

# ---------------------------------------------------------------------------
# 6. Same style of output as your original script
# ---------------------------------------------------------------------------
print("Emergency Vehicle Records:\n")
emergency_records = df[df["Vehicle_Type"] != "Normal"]
print(
    emergency_records[
        [
            "Vehicle_ID",
            "Vehicle_Type",
            "UAV_ID",
            "Signal_Strength",
            "Delay",
            "PDR",
            "Urgency",
            "Priority_Score",
            "Priority_Label",
        ]
    ].head()
)

print("\n======================================")
print(" Highest Priority Communication ")
print("======================================\n")

best_priority = df.loc[df["Priority_Score"].idxmax()]
print(
    best_priority[
        [
            "Vehicle_ID",
            "Vehicle_Type",
            "UAV_ID",
            "Signal_Strength",
            "Delay",
            "PDR",
            "Urgency",
            "Priority_Score",
            "Priority_Label",
        ]
    ]
)

print("\n======================================")
print(" Fuzzy Emergency Priority System Completed ")
print("======================================")