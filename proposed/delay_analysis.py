import matplotlib.pyplot as plt

from analysis_helpers import graph_path, iter_scenario_frames, load_dataset, sort_vehicle_index

df = load_dataset()

for sid, sname, sdf in iter_scenario_frames(df):
    avg_delay = sort_vehicle_index(sdf.groupby("Vehicle_ID")["Delay"].mean())

    print(f"\nAVERAGE DELAY — {sname} ({sid})\n")
    print(avg_delay)

    plt.figure(figsize=(7, 5))
    avg_delay.plot(kind="line", marker="o")
    plt.title(f"Average Communication Delay per Vehicle — {sname}")
    plt.xlabel("Vehicle ID")
    plt.ylabel("Average Delay (ms)")
    plt.grid(True)

    out = graph_path(sid, "average_delay_per_vehicle.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")
