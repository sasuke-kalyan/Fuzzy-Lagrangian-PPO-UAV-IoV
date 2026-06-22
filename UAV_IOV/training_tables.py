"""
Export episode training logs to Excel and publication-ready PNG tables.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from episode_training_callback import EpisodeTrainingLog
from results_paths import TRAINING_TABLES_DIR, training_table_png_path, training_xlsx_path


def export_training_table(log: EpisodeTrainingLog, preview_rows: int = 12) -> tuple[str, str]:
    """Write .xlsx (full) and .png (preview) for one scenario."""
    TRAINING_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(log.to_dataframe_rows())

    xlsx_path = training_xlsx_path(log.scenario_id)
    df.to_excel(xlsx_path, index=False, sheet_name="Training")

    png_path = _save_table_preview_image(df, log.scenario_id, preview_rows=preview_rows)
    return str(xlsx_path), str(png_path)


def _save_table_preview_image(
    df: pd.DataFrame,
    scenario_id: str,
    preview_rows: int = 12,
) -> str:
    """PNG table: first/last rows for readability with thousands of episodes."""
    if len(df) <= preview_rows * 2:
        display = df
        caption = f"All {len(df)} episodes"
    else:
        head = df.head(preview_rows)
        tail = df.tail(preview_rows)
        sep = pd.DataFrame([{c: "..." for c in df.columns}])
        display = pd.concat([head, sep, tail], ignore_index=True)
        caption = f"Preview: first {preview_rows} + last {preview_rows} of {len(df)} episodes"

    fig, ax = plt.subplots(figsize=(10, min(0.45 * len(display) + 1.5, 14)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title(
        f"Training Results — {scenario_id}\n{caption}",
        fontsize=12,
        fontweight="bold",
        pad=16,
    )
    out = training_table_png_path(scenario_id)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def export_all_training_tables(logs: dict[str, EpisodeTrainingLog]) -> list[tuple[str, str]]:
    outputs = []
    for sid, log in logs.items():
        if not log.records:
            continue
        xlsx, png = export_training_table(log)
        print(f"Saved training table: {xlsx}")
        print(f"Saved training table image: {png}")
        outputs.append((xlsx, png))
    return outputs
