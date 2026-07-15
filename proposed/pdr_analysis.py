import matplotlib.pyplot as plt

from analysis_helpers import graph_path, iter_scenario_frames, load_dataset, sort_vehicle_index

df = load_dataset()

for sid, sname, sdf in iter_scenario_frames(df):
    avg_pdr = sort_vehicle_index(sdf.groupby("Vehicle_ID")["PDR"].mean())

    print(f"\nAVERAGE PDR — {sname} ({sid})\n")
    print(avg_pdr)

    plt.figure(figsize=(7, 5))
    avg_pdr.plot(kind="line", marker="o")
    plt.title(f"Average PDR per Vehicle — {sname}")
    plt.xlabel("Vehicle ID")
    plt.ylabel("Average PDR")
    plt.grid(True)

    out = graph_path(sid, "average_pdr_per_vehicle.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")
