import math

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D


def no_discussion_voting_baseline_comparison_plot(
    result: pl.DataFrame,
    selected_temperature: float,
    selected_participants: set[int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    xlabel: str = "Improvement over best single-model (same participants)",
):

    plot_df = (
        result.filter(
            pl.col("method").is_in(["mixed_families", "mixed_sizes", "single_model"])
        )
        .filter(pl.col("temperature").is_close(selected_temperature, abs_tol=0.1))
        .filter(pl.col("no_participants").is_in(selected_participants))
        .with_columns(
            [
                # performance deltas
                (pl.col("accuracy_mean") - pl.col("best_single"))
                .alias("delta_single")
                .fill_null(0.0),
                (pl.col("accuracy_mean") - pl.col("best_baseline"))
                .alias("delta_baseline")
                .fill_null(0.0),
                # readable labels
                pl.col("families")
                .list.unique()
                .list.sort()
                .list.join("+")
                .alias("family_label"),
                pl.col("sizes")
                .list.unique()
                .list.sort()
                .list.join("+")
                .alias("size_label"),
                # number of mixed models
                pl.col("families").list.len().alias("n_models"),
            ]
        )
    )

    family_short = {
        "Qwen3": "Q3",
        "Qwen3.5": "Q35",
        "Ministral": "M",
    }

    def compact_family_label(fams):
        return "+".join(family_short[f] for f in fams)

    pdf = plot_df.with_columns(
        pl.col("families")
        .list.unique()
        .list.sort()
        .map_elements(
            compact_family_label,
            return_dtype=pl.String,
        )
        .alias("family_short")
    ).to_pandas()

    markers = {
        "mixed_families": "o",
        "mixed_sizes": "s",
        "single_model": "^",
    }

    family_categories = sorted(pdf["family_short"].unique())

    cmap = plt.get_cmap("tab10")

    family_colors = {fam: cmap(i % 10) for i, fam in enumerate(family_categories)}

    participant_values = sorted(pdf["no_participants"].unique())

    n_plots = len(participant_values)

    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 7 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    offset = 0.0025
    left_shift = 0.001
    combo_offsets = {
        "H": [(-left_shift, offset)],
        "M": [(offset, 0.000)],
        "L": [(-left_shift, -offset)],
        "H+M": [(-left_shift, offset), (offset - left_shift, 0.000)],
        "H+L": [(-left_shift, offset), (-left_shift, -offset)],
        "L+M": [(offset - left_shift, 0.000), (-left_shift, -offset)],
        "H+L+M": [
            (-left_shift, offset),
            (offset - left_shift, 0.000),
            (-left_shift, -offset),
        ],
    }

    for ax, no_participants in zip(axes, participant_values):
        sub_pdf = pdf[pdf["no_participants"] == no_participants]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        for method in sub_pdf["method"].unique():
            sub = sub_pdf[sub_pdf["method"] == method]

            colors = [family_colors[f] for f in sub["family_short"]]

            ax.scatter(
                sub["delta_single"],
                sub["delta_baseline"],
                c=colors,
                s=160,
                marker=markers[method],
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
            )

            for x, y, combo in zip(
                sub["delta_single"],
                sub["delta_baseline"],
                sub["size_label"],
            ):
                for dx, dy in combo_offsets.get(combo, []):
                    ax.scatter(
                        x + dx,
                        y + dy,
                        s=18,
                        color="#00000066",
                        # zorder=5,
                    )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="gray",
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
            color="gray",
        )

        ax.set_title(
            f"{no_participants} participants",
            fontsize=12,
        )

        ax.grid(alpha=0.3)

    for i in range(len(participant_values), len(axes)):
        fig.delaxes(axes[i])

    fig.supxlabel(
        xlabel,
        fontsize=14,
    )

    fig.supylabel(
        "Improvement over best baseline (1 part.)",
        fontsize=14,
    )

    fig.suptitle(
        f"No-Debate: Mixed Methods vs Best constituent Models (temperature {selected_temperature})",
        fontsize=16,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label=method.replace("_", " ").capitalize(),
        )
        for method, marker in markers.items()
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label=family,
        )
        for family, color in sorted(family_colors.items(), key=lambda x: len(x[0]))
    ]

    for ax in fig.axes:
        ax.add_artist(
            ax.legend(
                handles=method_handles,
                title="Method",
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
            )
        )
        ax.text(
            ax.get_xlim()[0] * 0.66,
            ax.get_ylim()[1] * 0.65,
            "Contained Sizes:\n• on top    = H\n• on right  = M\n• on bottom = L",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )
        ax.legend(
            handles=family_handles,
            title="Family",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.78),
        )

    fig.tight_layout()

    return fig
