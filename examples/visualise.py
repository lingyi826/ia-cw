"""Create an SVG figure and Markdown table from evaluation CSV data.

This module deliberately uses only the Python standard library so that the
coursework visualisations can be reproduced without installing plotting tools.
"""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

from examples.evaluate import AGENT_TYPES
from examples.train_qlearning import TASK_MODES


PLOTS = (
    ("task_completion_rate", "Task completion rate", True),
    ("office_cleanliness", "Mean normalised office cleanliness", True),
    ("norm_compliance_rate", "N3 work compliance rate", True),
    ("social_welfare", "Social welfare", False),
)
COLOURS = {
    "random": "#9AA0A6",
    "rulebased": "#2A6F97",
    "qlearning": "#E07A5F",
}


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["mode"], row["agent"]): row for row in rows}


def write_markdown_table(
    path: Path, rows: dict[tuple[str, str], dict[str, str]]
) -> None:
    lines = [
        "| Mode | Agent | Completion rate | Cleanliness | Compliance | Social welfare |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode in TASK_MODES:
        for agent in AGENT_TYPES:
            row = rows[mode, agent]

            def value(metric: str) -> str:
                mean_value = float(row[f"{metric}_mean"])
                std_value = float(row[f"{metric}_std"])
                return f"{mean_value:.3f} +/- {std_value:.3f}"

            lines.append(
                f"| {mode} | {agent} | {value('task_completion_rate')} | "
                f"{value('office_cleanliness')} | {value('norm_compliance_rate')} | "
                f"{value('social_welfare')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "middle",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#202124">{escape(text)}</text>'
    )


def add_chart(
    output: list[str],
    rows: dict[tuple[str, str], dict[str, str]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    metric: str,
    title: str,
    rate: bool,
) -> None:
    """Append one grouped bar chart with standard-deviation error bars."""
    plot_x = x + 62
    plot_y = y + 42
    plot_width = width - 82
    plot_height = height - 92

    values = [
        float(rows[mode, agent][f"{metric}_mean"])
        + float(rows[mode, agent][f"{metric}_std"])
        for mode in TASK_MODES
        for agent in AGENT_TYPES
    ]
    maximum = 1.0 if rate else max(values) * 1.10
    maximum = maximum or 1.0

    output.append(svg_text(x + width / 2, y + 22, title, size=17, weight="bold"))
    output.append(
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" '
        f'y2="{plot_y + plot_height}" stroke="#444" stroke-width="1"/>'
    )
    output.append(
        f'<line x1="{plot_x}" y1="{plot_y + plot_height}" '
        f'x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" '
        'stroke="#444" stroke-width="1"/>'
    )

    for tick in range(6):
        tick_value = maximum * tick / 5
        tick_y = plot_y + plot_height - plot_height * tick / 5
        output.append(
            f'<line x1="{plot_x}" y1="{tick_y:.1f}" '
            f'x2="{plot_x + plot_width}" y2="{tick_y:.1f}" '
            'stroke="#DADCE0" stroke-width="1"/>'
        )
        output.append(
            svg_text(
                plot_x - 8,
                tick_y + 5,
                f"{tick_value:.1f}",
                size=11,
                anchor="end",
            )
        )

    group_width = plot_width / len(TASK_MODES)
    bar_width = group_width * 0.22
    for mode_index, mode in enumerate(TASK_MODES):
        centre = plot_x + group_width * (mode_index + 0.5)
        output.append(svg_text(centre, plot_y + plot_height + 25, mode, size=12))
        for agent_index, agent in enumerate(AGENT_TYPES):
            row = rows[mode, agent]
            mean_value = float(row[f"{metric}_mean"])
            std_value = float(row[f"{metric}_std"])
            bar_x = centre + (agent_index - 1) * bar_width - bar_width * 0.42
            bar_height = plot_height * mean_value / maximum
            bar_y = plot_y + plot_height - bar_height
            output.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" '
                f'width="{bar_width * 0.84:.1f}" height="{bar_height:.1f}" '
                f'fill="{COLOURS[agent]}"/>'
            )

            error_top = plot_y + plot_height - plot_height * min(
                maximum, mean_value + std_value
            ) / maximum
            error_bottom = plot_y + plot_height - plot_height * max(
                0.0, mean_value - std_value
            ) / maximum
            error_x = bar_x + bar_width * 0.42
            output.append(
                f'<line x1="{error_x:.1f}" y1="{error_top:.1f}" '
                f'x2="{error_x:.1f}" y2="{error_bottom:.1f}" '
                'stroke="#111" stroke-width="1.3"/>'
            )
            output.append(
                f'<line x1="{error_x - 4:.1f}" y1="{error_top:.1f}" '
                f'x2="{error_x + 4:.1f}" y2="{error_top:.1f}" '
                'stroke="#111" stroke-width="1.3"/>'
            )


def write_svg(path: Path, rows: dict[tuple[str, str], dict[str, str]]) -> None:
    width, height = 1200, 850
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            32,
            "Agent comparison across task configurations",
            size=23,
            weight="bold",
        ),
    ]

    legend_x = width / 2 - 185
    for index, agent in enumerate(AGENT_TYPES):
        item_x = legend_x + index * 150
        output.append(
            f'<rect x="{item_x:.1f}" y="49" width="22" height="14" '
            f'fill="{COLOURS[agent]}"/>'
        )
        output.append(svg_text(item_x + 30, 61, agent, size=13, anchor="start"))

    chart_width, chart_height = 570, 350
    positions = ((20, 80), (610, 80), (20, 445), (610, 445))
    for (metric, title, rate), (x, y) in zip(PLOTS, positions, strict=True):
        add_chart(
            output,
            rows,
            x=x,
            y=y,
            width=chart_width,
            height=chart_height,
            metric=metric,
            title=title,
            rate=rate,
        )
    output.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/evaluation_summary.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/evaluation_comparison.svg")
    )
    parser.add_argument(
        "--table", type=Path, default=Path("results/evaluation_summary.md")
    )
    args = parser.parse_args()

    rows = read_summary(args.input)
    missing = [
        (mode, agent)
        for mode in TASK_MODES
        for agent in AGENT_TYPES
        if (mode, agent) not in rows
    ]
    if missing:
        raise ValueError(f"Evaluation summary is missing groups: {missing}")

    write_svg(args.output, rows)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_table(args.table, rows)
    print(
        json.dumps(
            {
                "type": "visualisation_files",
                "figure": str(args.output),
                "table": str(args.table),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
