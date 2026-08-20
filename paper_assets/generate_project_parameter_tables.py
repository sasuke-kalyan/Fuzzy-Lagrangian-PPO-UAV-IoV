"""Create publication-ready parameter-table PNGs from the active implementation.

The tables deliberately use source values from proposed/ rather than the older
report-export values.  Run from the repository root:
    MPLCONFIGDIR=/tmp/mpl python3 paper_assets/generate_project_parameter_tables.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).parent / "generated" / "parameter_tables"


TABLES = (
    (
        "table_1_experimental_configuration.png",
        "Table 1\nExperimental configuration of the proposed Fuzzy LP-PPO UAV-IoV system.",
        ["According to", "Attribute", "Value"],
        [
            ["Simulation infrastructure", "Operational area", "2,000 × 2,000 m²"],
            ["", "Candidate UAVs", "8"],
            ["", "UAV altitude", "80–200 m"],
            ["", "Communication range", "500 m"],
            ["", "Initial UAV energy", "30–50 internal units"],
            ["", "Energy drain per selected step", "1–3 internal units"],
            ["", "Observation dimension", "34"],
            ["", "Action space", "8 discrete UAV-selection actions"],
            ["QoS constraints", "Maximum communication delay", "95 ms"],
            ["", "Minimum packet delivery ratio", "55 %"],
            ["", "Minimum residual energy", "20 %"],
            ["", "Minimum signal strength", "0.10"],
            ["Adaptive Lagrangian", "Initial multipliers (D / PDR / E / S)", "0.35 / 0.20 / 0.60 / 0.35"],
            ["", "Multiplier step size", "0.03"],
            ["", "Multiplier range", "[0, 10]"],
        ],
        (0.23, 0.48, 0.29),
    ),
    (
        "table_2_ppo_hyperparameters.png",
        "Table 2\nFuzzy LP-PPO training hyperparameters.",
        ["Parameter", "Value"],
        [
            ["Learning algorithm", "PPO (MlpPolicy)"],
            ["Learning rate", "3 × 10⁻⁴"],
            ["Rollout steps (n_steps)", "200"],
            ["Minibatch size", "64 transitions"],
            ["PPO epochs per update", "10"],
            ["Discount factor (γ)", "0.99"],
            ["GAE parameter (λ)", "0.95"],
            ["PPO clipping parameter (ε)", "0.20"],
            ["Entropy coefficient", "0.02"],
            ["Actor network", "128–128 neurons"],
            ["Critic network", "128–128 neurons"],
            ["Training episodes", "100 per scenario"],
            ["Steps per episode", "50"],
            ["Fuzzy warm-start samples / epochs", "800 / 6"],
            ["Warm-start minibatch size", "64"],
            ["Random seed", "42 (training default)"],
        ],
        (0.48, 0.52),
    ),
    (
        "table_3_scenario_configuration.png",
        "Table 3\nEvaluation scenarios used by the proposed Fuzzy LP-PPO system.",
        ["Scenario", "Vehicles", "Mobility / load", "Evaluation focus"],
        [
            ["Urban canyon", "20", "60–100 km/h; high task rate", "Collision avoidance and throughput"],
            ["Suburban crossroads", "15", "20–50 km/h; medium task rate", "UAV energy efficiency"],
            ["Emergency response", "15", "30–70 km/h; low baseline task rate", "Dynamic incident adaptation"],
        ],
        (0.22, 0.12, 0.32, 0.34),
    ),
)


def render_table(filename: str, title: str, headers: list[str], rows: list[list[str]], widths: tuple[float, ...]) -> None:
    """Render one high-resolution, black-rule academic table."""
    fig_height = 1.25 + 0.38 * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(10.6, fig_height), dpi=300)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", family="serif", linespacing=1.4)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="center",
        colWidths=widths,
        bbox=[0.02, 0.02, 0.96, 0.79],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.8)
    table.scale(1, 1.25)

    cells = table.get_celld()
    nrows, ncols = len(rows), len(headers)
    for (row, col), cell in cells.items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_text_props(family="serif")
        if row == 0:
            cell.set_text_props(weight="bold", ha="center")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.1)
        elif row == nrows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.1)
        else:
            cell.visible_edges = ""
            cell.set_linewidth(0.0)
        if row > 0 and col == 0 and rows[row - 1][0]:
            cell.set_text_props(weight="normal")

    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        render_table(*table)
    print(f"Created {len(TABLES)} tables in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
