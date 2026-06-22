import matplotlib.pyplot as plt

from analysis_helpers import graph_path, iter_scenario_frames, load_dataset, sort_vehicle_index

df = load_dataset()

for sid, sname, sdf in iter_scenario_frames(df):
    sdf = sdf.copy()
    sdf["Reliability_Score"] = (
        (sdf["Signal_Strength"] * 50)
        + (sdf["PDR"] * 0.4)
        - (sdf["Delay"] * 0.5)
    )
    avg_reliability = sort_vehicle_index(
        sdf.groupby("Vehicle_ID")["Reliability_Score"].mean()
    )

    print(f"\nRELIABILITY SCORE — {sname} ({sid})\n")
    print(avg_reliability)

    plt.figure(figsize=(7, 5))
    avg_reliability.plot(kind="line", marker="o")
    plt.title(f"Average Reliability Score — {sname}")
    plt.xlabel("Vehicle ID")
    plt.ylabel("Reliability Score")
    plt.grid(True)

    out = graph_path(sid, "average_reliability_score.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")
