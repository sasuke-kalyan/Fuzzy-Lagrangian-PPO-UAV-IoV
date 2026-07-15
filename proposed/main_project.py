import pandas as pd

from data_loader import iter_scenario_frames, load_dataset
from network_config import NUM_UAVS, SCENARIO_IDS

df = load_dataset()

print("\n==========================================")
print("   UAV-IoV Communication System Analysis")
print("   (3 Paper Simulation Scenarios)")
print("==========================================\n")

print("Configured scenarios:", ", ".join(SCENARIO_IDS))
print("Dataset rows:", len(df))
if "Scenario_ID" in df.columns:
    print("Scenarios in data:", df["Scenario_ID"].unique().tolist())

for sid, sname, sdf in iter_scenario_frames(df):
    print("\n------------------------------------------")
    print(f"  Scenario: {sname} ({sid})")
    print("------------------------------------------\n")

    print("Dataset Preview:\n")
    print(sdf.head(3))

    total_vehicles = sdf["Vehicle_ID"].nunique()
    total_uavs = sdf["UAV_ID"].nunique()
    print("\nTotal Vehicles :", total_vehicles)
    print("Total UAVs     :", total_uavs, f"(configured: {NUM_UAVS})")
    print("Timesteps      :", sdf["Timestamp"].nunique())
    print("Link records   :", len(sdf))

    print("\n Average Communication Metrics \n")
    print("Average Distance          :", round(sdf["Distance"].mean(), 2))
    print("Average Signal Strength   :", round(sdf["Signal_Strength"].mean(), 2))
    print("Average Delay             :", round(sdf["Delay"].mean(), 2))
    print("Average PDR               :", round(sdf["PDR"].mean(), 2))
    print("Average Energy            :", round(sdf["Energy"].mean(), 2))

    sdf = sdf.copy()
    sdf["Reliability_Score"] = (
        (sdf["Signal_Strength"] * 50)
        + (sdf["PDR"] * 0.5)
        - (sdf["Delay"] * 0.4)
    )
    print("Average Reliability Score :", round(sdf["Reliability_Score"].mean(), 2))

    sdf["Reward"] = (sdf["Signal_Strength"] * 100) + sdf["PDR"] - sdf["Delay"]
    print("Average RL Reward         :", round(sdf["Reward"].mean(), 2))

    if "Emergency_Phase" in sdf.columns and sdf["Emergency_Phase"].any():
        pre = sdf[sdf["Emergency_Phase"] == 0]
        post = sdf[sdf["Emergency_Phase"] == 1]
        print("\n Emergency phase split:")
        print("  Pre-incident  delay:", round(pre["Delay"].mean(), 2), "ms")
        print("  Post-incident delay:", round(post["Delay"].mean(), 2), "ms")

    best = sdf.loc[sdf["PDR"].idxmax()]
    print("\n Best Communication Record ")
    print("Vehicle :", best["Vehicle_ID"])
    print("UAV     :", best["UAV_ID"])
    print("PDR     :", best["PDR"])

print("\n==========================================")
print(" Analysis modules (run separately):")
print("==========================================\n")
print("  dataset_generation.py       — regenerate all 3 scenario CSVs")
print("  train_ppo.py / train_scenario_rl.py — episode-based RL (default 1000 ep.)")
print("  run_training_and_evaluation.py — train + results/ graphs + Excel tables")
print("  scenario_analysis.py          — cross-scenario summary & plots")
print("  evaluation_pipeline.py        — results/ metric graphs + legacy graph_outputs")
print("  pdr_analysis.py / delay_analysis.py / reliability_score.py")
print("\n✔ Multi-scenario UAV-IoV analysis ready.\n")
