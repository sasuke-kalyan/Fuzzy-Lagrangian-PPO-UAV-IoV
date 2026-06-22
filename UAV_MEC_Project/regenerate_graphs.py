import json
from visualization.plots import plot_learning_curve

# MADDPG
with open("results/maddpg/training_log.json", "r") as f:
    rows = json.load(f)

plot_learning_curve(
    rows,
    "results/maddpg/learning_curve.png",
    "MADDPG"
)

# LC-MAPO
with open("results/lc_mapo/training_log.json", "r") as f:
    rows = json.load(f)

plot_learning_curve(
    rows,
    "results/lc_mapo/learning_curve.png",
    "LC-MAPO"
)

print("Graphs regenerated successfully!")