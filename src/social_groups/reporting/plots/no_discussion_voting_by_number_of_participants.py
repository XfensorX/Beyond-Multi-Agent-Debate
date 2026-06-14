import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def no_discussion_voting_by_number_of_participants(
    no_discussion_with_different_no_agents,
    title_base: str = "Voting Accuracy (No Debate)",
    figsize: tuple = (16, 7.8),
    alpha_band: float = 0.12,  # Lightness of the min-max band
):
    # === Data Preparation ===
    df_pl = no_discussion_with_different_no_agents.with_columns(
        pl.col("temperature").cast(pl.Float64).round(1)
    )

    # Aggregate across seeds: mean + min + max
    agg_df = (
        df_pl.group_by(["family", "parameters", "temperature", "no_participants"])
        .agg(
            pl.col("accuracy").mean().alias("accuracy_mean"),
            pl.col("accuracy").min().alias("accuracy_min"),
            pl.col("accuracy").max().alias("accuracy_max"),
            pl.col("accuracy").count().alias("n_seeds"),
        )
        .sort("family", "parameters", "temperature", "no_participants")
        .to_pandas()
    )

    agg_df["temperature"] = agg_df["temperature"].astype(str)

    # Create model label and size rank for consistent coloring
    families = sorted(agg_df["family"].unique())

    size_order = {}
    for fam in families:
        fam_params = sorted(agg_df[agg_df["family"] == fam]["parameters"].unique())
        size_order[fam] = {p: i for i, p in enumerate(fam_params)}

    agg_df["size_rank"] = agg_df.apply(
        lambda row: size_order[row["family"]][row["parameters"]], axis=1
    )

    # === Plot Settings ===
    sns.set_style("whitegrid", {"grid.linestyle": ":"})
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2.3,
            "lines.markersize": 7,
        }
    )

    size_palette = sns.color_palette(
        "muted", n_colors=len(agg_df["size_rank"].unique())
    )  # Change to "viridis", "crest", etc. if desired

    # Temperature styles
    temps = sorted(agg_df["temperature"].unique())
    markers = ["o", "s", "D", "^", "v"][: len(temps)]
    dashes = [(1, 0), (4, 1.5), (1, 3), (3, 1, 1, 1), (2, 1, 1, 1)][: len(temps)]
    temp_style_map = dict(zip(temps, zip(markers, dashes)))

    ncols = min(3, len(families))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize[0], figsize[1] * nrows),
        dpi=300,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, fam in zip(axes, families):
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks(np.arange(0.2, 1.01, 0.1))

        fam_df = agg_df[agg_df["family"] == fam].copy()

        for temp in temps:
            temp_df = fam_df[fam_df["temperature"] == temp]
            marker, dash = temp_style_map[temp]

            for size_rank in sorted(temp_df["size_rank"].unique()):
                sub = temp_df[temp_df["size_rank"] == size_rank]
                color = size_palette[size_rank]
                param_count = sub["parameters"].iloc[0]

                # Very light min-max band
                ax.fill_between(
                    sub["no_participants"],
                    sub["accuracy_min"],
                    sub["accuracy_max"],
                    color=color,
                    alpha=alpha_band,
                    linewidth=0,
                )

                # Main mean line with markers
                ax.plot(
                    sub["no_participants"],
                    sub["accuracy_mean"],
                    marker=marker,
                    dashes=dash,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    linewidth=2.3,
                    markersize=7,
                    label=(
                        f"{param_count:5.1f} B"
                        if isinstance(param_count, int)
                        else param_count
                    )
                    if temp == temps[0]
                    else None,
                )

        ax.set_title(f"{fam}", fontsize=13, pad=12, fontweight="semibold")
        ax.set_xlabel("Number of Participants", fontsize=12, labelpad=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if ax == axes[0]:
            ax.set_ylabel("Accuracy", fontsize=12, labelpad=10)
        else:
            ax.set_ylabel("")

        # Per-family parameter legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                title="Models",
                loc="lower right",
                frameon=True,
                fancybox=True,
                fontsize=9.5,
            )

        sns.despine(ax=ax, trim=True)

    for ax in axes[len(families) :]:
        ax.set_visible(False)

    # === Shared Temperature Legend (top-right) ===
    temp_handles = []
    for temp in temps:
        marker, dash = temp_style_map[temp]
        line = plt.Line2D(
            [],
            [],
            color="black",
            marker=marker,
            dashes=dash,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.2,
            label=f"{temp}",
        )
        temp_handles.append(line)

    fig.legend(
        temp_handles,
        [f"{t}" for t in temps],
        title="Temperature",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.96),
        frameon=True,
        fancybox=True,
        fontsize=10,
        title_fontsize=11,
    )

    fig.suptitle(title_base, fontsize=15, fontweight="semibold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
