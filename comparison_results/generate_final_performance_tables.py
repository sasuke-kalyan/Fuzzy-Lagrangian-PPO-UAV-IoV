from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "comparison_results" / "fair_qos_comparison.csv"
POLICY_CSV_PATH = ROOT / "UAV_IOV" / "graph_outputs" / "scenario_ppo_policy_comparison.csv"
DATASET_CSV_PATH = ROOT / "UAV_IOV" / "uav_iov_dataset.csv"
OUT_DIR = ROOT / "comparison_results" / "final performances"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_METHOD = "LC-MAPO"
PROPOSED_METHOD = "Fuzzy LP-PPO"

SCENARIO_LABELS = {
    "urban": "Urban Canyon",
    "suburban": "Suburban Crossroads",
    "emergency": "Emergency Response",
}

METRICS = [
    {
        "key": "avg_delay_ms",
        "label": "Average Delay (ms)",
        "short": "average_delay",
        "better": "lower",
        "fmt": "{:.2f}",
        "unit": "ms",
    },
    {
        "key": "avg_pdr_pct",
        "label": "PDR / Task Completion Rate (%)",
        "short": "pdr_task_completion",
        "better": "higher",
        "fmt": "{:.2f}",
        "unit": "%",
    },
    {
        "key": "avg_signal",
        "label": "Average Signal Strength",
        "short": "signal_quality",
        "better": "higher",
        "fmt": "{:.4f}",
        "unit": "",
    },
    {
        "key": "avg_energy_pct",
        "label": "Energy Consumption",
        "short": "energy_consumption",
        "better": "lower",
        "fmt": "{:.2f}",
        "unit": "",
    },
    {
        "key": "shared_qos_score",
        "label": "Overall QoS Score",
        "short": "overall_qos_score",
        "better": "higher",
        "fmt": "{:.2f}",
        "unit": "",
    },
]


def fmt_value(value, spec):
    return spec["fmt"].format(float(value))


def change_text(base, proposed, spec):
    base = float(base)
    proposed = float(proposed)
    delta = proposed - base

    if spec["better"] == "lower":
        gain = base - proposed
        direction = "lower" if gain >= 0 else "higher"
        unit = f" {spec['unit']}" if spec["unit"] else ""
        return f"{abs(gain):.2f}{unit} {direction}"

    direction = "higher" if delta >= 0 else "lower"
    unit = f" {spec['unit']}" if spec["unit"] else ""
    return f"{abs(delta):.2f}{unit} {direction}"


def result_text(base, proposed, spec):
    base = float(base)
    proposed = float(proposed)

    if spec["better"] == "lower":
        return "Proposed better" if proposed < base else "Base paper better"

    return "Proposed better" if proposed > base else "Base paper better"


def markdown_table(df):
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines]) + "\n"


def wrap_cell(value, width):
    if width is None:
        return value
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def save_table_files(
    df,
    stem,
    title,
    png_width=14,
    png_height=6.5,
    font_size=9,
    wrap_chars=None,
    col_widths=None,
):
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    png_path = OUT_DIR / f"{stem}.png"

    df.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n{markdown_table(df)}", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(png_width, png_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)

    display_df = df.apply(lambda col: col.map(lambda value: wrap_cell(value, wrap_chars)))

    table = ax.table(
        cellText=display_df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.88],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.2)
            cell.set_facecolor("#f2f2f2")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {png_path}")


def save_table_v_paper_style(table_df):
    stem = "table_v_style_project_kpi_comparison"
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    png_path = OUT_DIR / f"{stem}.png"

    table_df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Table V-Style KPI Comparison for Base Paper and Proposed Method\n\n"
        + markdown_table(table_df),
        encoding="utf-8",
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(11.2, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "TABLE V",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.935,
        "COMPARISON OF KEY PERFORMANCE INDICATORS FOR BASE PAPER AND PROPOSED METHOD",
        ha="center",
        va="center",
        fontsize=9,
        fontvariant="small-caps",
    )

    x_scenario = 0.13
    x_kpi = 0.30
    x_base = 0.67
    x_proposed = 0.85
    left = 0.03
    right = 0.97

    y_top = 0.89
    row_h = 0.034
    header_y = y_top - 0.025

    ax.hlines(y_top, left, right, color="black", linewidth=1.0)
    ax.text(x_scenario, header_y, "Scenario", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x_kpi, header_y, "Key Performance Indicators", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(x_base, header_y, "LC-MAPO", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x_proposed, header_y, "Fuzzy LP-PPO", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.hlines(y_top - 0.047, left, right, color="black", linewidth=0.75)

    y = y_top - 0.074
    scenarios = ["Urban Canyon", "Suburban Crossroads", "Emergency Response"]
    metric_order = [
        "Task Completion Rate (%)",
        "System Throughput (tasks/min)",
        "Average Task Service Time (s)",
        "Total Collisions (per episode)",
        "Total Low-Battery Events (per episode)",
        "Energy Cost per Task (kJ/task)",
        "Average Flight Distance (km/drone)",
    ]

    for scenario in scenarios:
        scenario_df = table_df[table_df["Scenario"] == scenario].set_index("Base Paper KPI")
        scenario_start_y = y

        for idx, metric in enumerate(metric_order):
            row = scenario_df.loc[metric]
            ax.text(x_kpi, y, metric, ha="left", va="center", fontsize=9.2)
            ax.text(x_base, y, row["Base Paper (LC-MAPO)"], ha="center", va="center", fontsize=9.2)
            ax.text(x_proposed, y, row["Proposed (Fuzzy LP-PPO)"], ha="center", va="center", fontsize=9.2)

            if idx in [2, 4]:
                ax.hlines(y - row_h / 2, 0.25, right, color="black", linewidth=0.45, alpha=0.85)
            y -= row_h

        scenario_mid_y = (scenario_start_y + y + row_h) / 2
        ax.text(
            x_scenario,
            scenario_mid_y,
            scenario,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.hlines(y + row_h / 2 - 0.002, left, right, color="black", linewidth=0.75)
        y -= 0.012

    ax.text(
        left + 0.02,
        y + 0.002,
        "* The values are shown using base-paper KPIs and closest available proposed-project indicators.",
        ha="left",
        va="top",
        fontsize=8,
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {png_path}")


def save_table_i_paper_style(thematic_df):
    stem = "thematic_classification_project_table"
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    png_path = OUT_DIR / f"{stem}.png"

    thematic_df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Thematic Classification of Task Offloading Literature with Proposed Method\n\n"
        + markdown_table(thematic_df),
        encoding="utf-8",
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(11.2, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "TABLE I", ha="center", va="center", fontsize=10)
    ax.text(
        0.5,
        0.95,
        "THEMATIC CLASSIFICATION OF TECHNICAL LITERATURE ON TASK OFFLOADING IN IOV",
        ha="center",
        va="center",
        fontsize=9,
        fontvariant="small-caps",
    )

    left = 0.03
    right = 0.97
    x_category = 0.10
    x_method = 0.29
    x_focus = 0.43
    x_ref = 0.92

    y_top = 0.91
    row_h = 0.073
    header_y = y_top - 0.023

    ax.hlines(y_top, left, right, color="black", linewidth=1.0)
    ax.text(x_category, header_y, "Category", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x_method, header_y, "Method / Technology", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(x_focus, header_y, "Focus / Problem Solved", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(x_ref, header_y, "Reference", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.hlines(y_top - 0.044, left, right, color="black", linewidth=0.75)

    y = y_top - 0.077
    row_positions = []
    for idx, row in thematic_df.iterrows():
        focus = wrap_cell(row["Focus / Problem Solved"], 46)
        ax.text(x_focus, y, focus, ha="left", va="center", fontsize=8.8)
        ax.text(x_ref, y, row["Reference"], ha="center", va="center", fontsize=8.8)
        row_positions.append((idx, y))
        y -= row_h

    for category, group in thematic_df.groupby("Category", sort=False):
        idxs = list(group.index)
        ys = [pos_y for idx, pos_y in row_positions if idx in idxs]
        ax.text(x_category, sum(ys) / len(ys), category, ha="center", va="center", fontsize=9.2)
        ax.hlines(min(ys) - row_h / 2, left, right, color="black", linewidth=0.65)

    for (category, method), group in thematic_df.groupby(["Category", "Method / Technology"], sort=False):
        idxs = list(group.index)
        ys = [pos_y for idx, pos_y in row_positions if idx in idxs]
        ax.text(x_method, sum(ys) / len(ys), method, ha="center", va="center", fontsize=8.9)
        if len(ys) > 1:
            ax.hlines(min(ys) - row_h / 2, 0.22, right, color="black", linewidth=0.45, alpha=0.85)

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {png_path}")


def get_value(df, scenario, method, metric):
    row = df[(df["scenario"] == scenario) & (df["method"] == method)]
    if row.empty:
        raise ValueError(f"Missing row for scenario={scenario}, method={method}")
    return float(row.iloc[0][metric])


def build_tables():
    df = pd.read_csv(CSV_PATH)

    main_rows = []
    metric_rows = {metric["short"]: [] for metric in METRICS}

    for scenario in ["urban", "suburban", "emergency"]:
        scenario_label = SCENARIO_LABELS[scenario]

        for metric in METRICS:
            base = get_value(df, scenario, BASE_METHOD, metric["key"])
            proposed = get_value(df, scenario, PROPOSED_METHOD, metric["key"])
            base_text = fmt_value(base, metric)
            proposed_text = fmt_value(proposed, metric)
            change = change_text(base, proposed, metric)
            result = result_text(base, proposed, metric)

            main_rows.append(
                {
                    "Scenario": scenario_label,
                    "Key Performance Indicator": metric["label"],
                    "Base Paper (LC-MAPO)": base_text,
                    "Proposed (Fuzzy LP-PPO)": proposed_text,
                    "Change vs Base": change,
                    "Result": result,
                }
            )

            metric_rows[metric["short"]].append(
                {
                    "Scenario": scenario_label,
                    "Base Paper (LC-MAPO)": base_text,
                    "Proposed (Fuzzy LP-PPO)": proposed_text,
                    "Change vs Base": change,
                    "Result": result,
                }
            )

    main_df = pd.DataFrame(main_rows)
    save_table_files(
        main_df,
        "final_performance_comparison_table",
        "Final Performance Comparison Across Three Simulation Scenarios",
        png_width=16,
        png_height=9,
        font_size=8,
    )

    for metric in METRICS:
        metric_df = pd.DataFrame(metric_rows[metric["short"]])
        save_table_files(
            metric_df,
            f"{metric['short']}_table",
            metric["label"],
            png_width=11,
            png_height=3.6,
            font_size=9,
        )

    thematic_rows = [
        {
            "Category": "V2V Task Offloading",
            "Method / Technology": "Game Theory",
            "Focus / Problem Solved": "Incentive mechanism and cooperative resource sharing",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "V2V Task Offloading",
            "Method / Technology": "Learning-Based",
            "Focus / Problem Solved": "Server selection, trust-aware sharing, and proactive caching",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "V2I Task Offloading",
            "Method / Technology": "Learning-Based",
            "Focus / Problem Solved": "Energy efficiency, resource management, and task caching",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "V2I Task Offloading",
            "Method / Technology": "Optimization",
            "Focus / Problem Solved": "Joint offloading and resource allocation",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "V2D Task Offloading",
            "Method / Technology": "Optimization",
            "Focus / Problem Solved": "Drone deployment, path planning, and resource scheduling",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "V2D Task Offloading",
            "Method / Technology": "Learning-Based",
            "Focus / Problem Solved": "Drone positioning, user association, trajectory control, and cost minimization",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "V2D Task Offloading",
            "Method / Technology": "Fuzzy LP-PPO",
            "Focus / Problem Solved": "Proposed reliable UAV-IoV communication with delay, energy, PDR, signal, and QoS optimization",
            "Reference": "Proposed Work",
        },
        {
            "Category": "Hybrid & Cross-Layer",
            "Method / Technology": "Hybrid",
            "Focus / Problem Solved": "Joint scheduling, adaptive offloading, and delay reduction",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "Hybrid & Cross-Layer",
            "Method / Technology": "Cross-Layer",
            "Focus / Problem Solved": "Air-ground cost minimization, resource allocation, and trajectory optimization",
            "Reference": "Base Paper Survey",
        },
        {
            "Category": "Hybrid & Cross-Layer",
            "Method / Technology": "Fuzzy Lagrangian PPO",
            "Focus / Problem Solved": "Proposed cross-metric policy optimization for reliable dynamic UAV-IoV networks",
            "Reference": "Proposed Work",
        },
    ]

    thematic_df = pd.DataFrame(thematic_rows)
    save_table_i_paper_style(thematic_df)

    base_table_v = {
        "Urban Canyon": {
            "Task Completion Rate (%)": "92.31 +/- 3.5",
            "System Throughput (tasks/min)": "15.38 +/- 0.6",
            "Average Task Service Time (s)": "11.23 +/- 1.5",
            "Total Collisions (per episode)": "0.1 +/- 0.2",
            "Total Low-Battery Events (per episode)": "0.0 +/- 0.0",
            "Energy Cost per Task (kJ/task)": "1.23 +/- 0.15",
            "Average Flight Distance (km/drone)": "3.01 +/- 0.25",
        },
        "Suburban Crossroads": {
            "Task Completion Rate (%)": "95.12 +/- 2.8",
            "System Throughput (tasks/min)": "7.93 +/- 0.2",
            "Average Task Service Time (s)": "25.81 +/- 2.1",
            "Total Collisions (per episode)": "0.0 +/- 0.0",
            "Total Low-Battery Events (per episode)": "0.1 +/- 0.3",
            "Energy Cost per Task (kJ/task)": "2.53 +/- 0.28",
            "Average Flight Distance (km/drone)": "5.12 +/- 0.41",
        },
        "Emergency Response": {
            "Task Completion Rate (%)": "88.93 +/- 4.2",
            "System Throughput (tasks/min)": "9.88 +/- 0.5",
            "Average Task Service Time (s)": "15.62 +/- 1.8",
            "Total Collisions (per episode)": "0.2 +/- 0.4",
            "Total Low-Battery Events (per episode)": "0.3 +/- 0.5",
            "Energy Cost per Task (kJ/task)": "1.83 +/- 0.22",
            "Average Flight Distance (km/drone)": "4.01 +/- 0.35",
        },
    }

    policy_df = pd.read_csv(POLICY_CSV_PATH)
    dataset_df = pd.read_csv(DATASET_CSV_PATH)
    table_v_rows = []

    metric_mapping = [
        (
            "Task Completion Rate (%)",
            "Task Completion Rate / PDR (%)",
            lambda row, _sdf: f"{row['Avg_PDR_pct']:.2f}",
        ),
        (
            "System Throughput (tasks/min)",
            "System Throughput Proxy",
            lambda row, _sdf: f"{row['Avg_PDR_pct'] * row['Avg_Signal']:.2f}",
        ),
        (
            "Average Task Service Time (s)",
            "Average Delay (ms)",
            lambda row, _sdf: f"{row['Avg_Delay_ms']:.2f}",
        ),
        (
            "Total Collisions (per episode)",
            "Out-of-Range Risk (%)",
            lambda row, _sdf: f"{row['Out_Of_Range_Rate_pct']:.2f}",
        ),
        (
            "Total Low-Battery Events (per episode)",
            "Weak-Signal Risk (%)",
            lambda row, _sdf: f"{row['Weak_Signal_Risk_pct']:.2f}",
        ),
        (
            "Energy Cost per Task (kJ/task)",
            "Average Energy",
            lambda row, _sdf: f"{row['Avg_Energy']:.2f}",
        ),
        (
            "Average Flight Distance (km/drone)",
            "Average Link Distance (km)",
            lambda _row, sdf: f"{sdf['Distance'].mean() / 1000.0:.2f}",
        ),
    ]

    for scenario_id, scenario_label in [
        ("urban_canyon", "Urban Canyon"),
        ("suburban_crossroads", "Suburban Crossroads"),
        ("emergency_response", "Emergency Response"),
    ]:
        proposed_row = policy_df[
            (policy_df["Scenario_ID"] == scenario_id)
            & (policy_df["Method"] == "Trained PPO Model")
        ].iloc[0]
        scenario_dataset = dataset_df[dataset_df["Scenario_ID"] == scenario_id]

        for base_metric, project_metric, proposed_getter in metric_mapping:
            table_v_rows.append(
                {
                    "Scenario": scenario_label,
                    "Base Paper KPI": base_metric,
                    "Project KPI Used": project_metric,
                    "Base Paper (LC-MAPO)": base_table_v[scenario_label][base_metric],
                    "Proposed (Fuzzy LP-PPO)": proposed_getter(proposed_row, scenario_dataset),
                }
            )

    table_v_df = pd.DataFrame(table_v_rows)
    save_table_v_paper_style(table_v_df)

    readme = """# Final Performances

These tables follow the comparison style of Table V in `base paper 1.pdf`.

Only two methods are presented:

- Base Paper: LC-MAPO
- Proposed: Fuzzy LP-PPO

The source data is `comparison_results/fair_qos_comparison.csv`.
Each table is provided as CSV, Markdown, and PNG for direct use in reports or presentations.

The thematic classification table follows the style of Table I in `base paper 1.pdf`
and adds the proposed Fuzzy LP-PPO method under the project-relevant V2D and
cross-layer UAV-IoV categories.

The Table V-style KPI table keeps the base-paper LC-MAPO values and maps the
available project outputs to the closest corresponding Fuzzy LP-PPO indicators.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved: {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    build_tables()
