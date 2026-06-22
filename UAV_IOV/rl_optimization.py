"""
RL analysis: Lagrangian batch optimization + neural PPO (if trained).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_dataset
import matplotlib.pyplot as plt

from lagrangian_solver import base_reward, run_primal_dual

MODEL_PATH = Path("models/ppo_uav_iov")
GAT_MODEL_PATH = Path("models/gat_uav_iov.pt")


def _run_lagrangian_section(df: pd.DataFrame) -> pd.DataFrame:
    print("\n======================================")
    print(" Lagrangian RL Reward Optimization ")
    print("======================================\n")

    df = df.copy()
    df["Reward"] = df.apply(base_reward, axis=1)

    df_lag, final_lambda = run_primal_dual(df, objective_col="Reward")

    print("\nFinal multipliers:")
    print(final_lambda)

    print("\nSample rows (Reward vs Lagrangian_Score):\n")
    print(
        df_lag[
            [
                "Vehicle_ID",
                "UAV_ID",
                "Delay",
                "PDR",
                "Energy",
                "Reward",
                "Lagrangian_Score",
                "Violation_Delay",
                "Violation_PDR",
                "Violation_Energy",
            ]
        ].head(10)
    )

    avg_reward = df_lag.groupby("Vehicle_ID")["Reward"].mean()
    avg_lag = df_lag.groupby("Vehicle_ID")["Lagrangian_Score"].mean()

    # Sort vehicle IDs in numerical order (V1, V2, ..., V10)
    avg_reward = avg_reward.reindex(sorted(avg_reward.index, key=lambda x: int(x[1:])))
    avg_lag = avg_lag.reindex(sorted(avg_lag.index, key=lambda x: int(x[1:])))

    print("\nAverage unconstrained Reward per vehicle:\n")
    print(avg_reward)
    print("\nAverage Lagrangian_Score per vehicle:\n")
    print(avg_lag)

    plt.figure(figsize=(7, 5))
    avg_lag.plot(kind='line', marker='o')
    plt.title("Average Lagrangian Score per Vehicle")
    plt.xlabel("Vehicle ID")
    plt.ylabel("Lagrangian Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lagrangian_score_per_vehicle.png", dpi=120)
    print("\nSaved plot: lagrangian_score_per_vehicle.png")
    plt.close()

    return df_lag


def _run_ppo_section() -> None:
    print("\n======================================")
    print(" Neural PPO (Proximal Policy Optimization) ")
    print("======================================\n")

    zip_path = MODEL_PATH.with_suffix(".zip")
    if not zip_path.exists() and not MODEL_PATH.exists():
        print("No trained PPO model found.")
        print("  Train:  python train_ppo.py")
        print("  Eval:   python evaluate_ppo.py")
        return

    from evaluate_ppo import main as evaluate_main

    evaluate_main()


def _run_gat_section() -> None:
    print("\n======================================")
    print(" Graph Attention Network (GAT) ")
    print("======================================\n")

    if not GAT_MODEL_PATH.exists():
        print("No trained GAT model found.")
        print("  Train:  python train_gat.py")
        print("  Eval:   python evaluate_gat.py")
        print("  Route:  python gat_routing.py")
        return

    from evaluate_gat import main as evaluate_gat_main

    evaluate_gat_main()


def main():
    df = load_dataset()
    _run_lagrangian_section(df)
    _run_ppo_section()
    _run_gat_section()
    print("\nRL optimization completed.\n")


if __name__ == "__main__":
    main()
