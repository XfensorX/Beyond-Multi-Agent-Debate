import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def accuracy_change_by_group_constellation(
    changed_prompt_mad,
    only_changed_prompt,
):

    individual_answer_col = "___parsed_individual_answers_before___"

    # -----------------------------
    # Build individual-level rows
    # -----------------------------

    individual_df = (
        changed_prompt_mad.with_columns(
            [
                # Split constellation string into role labels: "LMH" -> ["L", "M", "H"]
                pl.col("group_constellation").str.split("").alias("role_list")
            ]
        )
        .explode([individual_answer_col, "role_list"])
        .rename(
            {
                individual_answer_col: "individual_answer",
                "role_list": "individual_role",
            }
        )
        .with_columns(
            [
                (pl.col("individual_answer") == pl.col("answer_string"))
                .cast(pl.Float64)
                .alias("individual_is_correct")
            ]
        )
    )

    # -----------------------------
    # Question-balanced baseline
    # -----------------------------

    single_model_baselines = (
        individual_df
        # .drop_nulls(["individual_answer", "answer_string", "individual_role"])
        .group_by("model_family", "individual_role", "question_id")
        .agg(
            pl.mean("individual_is_correct").alias("question_role_accuracy"),
            pl.len().alias("XXX"),
        )
        .group_by("model_family", "individual_role")
        .agg(
            [
                pl.mean("question_role_accuracy").alias("single_model_baseline"),
                pl.len().alias("n_questions"),
                pl.col("XXX").unique().alias("n_unique_answers_per_question"),
            ]
        )
        .sort(["model_family", "individual_role"])
    )

    # -----------------------------
    # Configuration
    # -----------------------------

    family_col = "model_family"
    group_col = "group_constellation"

    before_col = "accuracy_before"
    after_col = "accuracy"
    increase_col = "accuracy_increase"

    # -----------------------------
    # Aggregate to one row per family × constellation
    # -----------------------------

    plot_base = (
        only_changed_prompt.select(
            family_col,
            group_col,
            before_col,
            after_col,
            increase_col,
        ).drop_nulls([family_col, group_col, before_col, after_col])
        # .group_by(family_col, group_col)
        # .agg(
        #     [
        #         pl.mean(before_col).alias(before_col),
        #         pl.mean(after_col).alias(after_col),
        #         pl.mean(increase_col).alias(increase_col),
        #     ]
        # )
    )

    # -----------------------------
    # Shared y-order based on within-family ranks
    # Best average rank at top
    # -----------------------------

    rank_df = (
        plot_base.with_columns(
            pl.col(after_col)
            .rank(method="average", descending=True)
            .over(family_col)
            .alias("family_rank")
        )
        .group_by(group_col)
        .agg(pl.mean("family_rank").alias("mean_family_rank"))
        .sort("mean_family_rank", descending=False)
    )

    group_order = rank_df.get_column(group_col).to_list()

    # Matplotlib y positions: top group gets highest y
    y_positions = {
        group: len(group_order) - 1 - i for i, group in enumerate(group_order)
    }

    # -----------------------------
    # Plot styling
    # -----------------------------

    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="muted",
        font_scale=0.95,
    )

    families = sorted(plot_base.get_column(family_col).unique().to_list())
    palette = sns.color_palette("muted", n_colors=len(families))
    family_colors = dict(zip(families, palette))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(families),
        figsize=(15, max(7, 0.35 * len(group_order))),
        sharey=True,
        sharex=False,
    )

    if len(families) == 1:
        axes = [axes]

        # Draw Basline:

    baseline_pdf = single_model_baselines.to_pandas()

    role_styles = {
        "L": {"linestyle": "--"},
        "M": {"linestyle": "-."},
        "H": {"linestyle": ":"},
    }

    # Build display labels

    label_df = only_changed_prompt.group_by("group_constellation").agg(
        pl.mean("call_cost_heuristic").alias("cost")
    )

    label_map = {
        row["group_constellation"]: f"{row['group_constellation']:>3s}"
        for row in label_df.to_dicts()
    }

    for ax, family in zip(axes, families):
        family_baselines = baseline_pdf[baseline_pdf["model_family"] == family]

        color = family_colors[family]

        for _, row in family_baselines.iterrows():
            role = row["individual_role"]

            baseline = row["single_model_baseline"]

            ax.axvline(
                baseline,
                color=color,
                linestyle=role_styles.get(role, {}).get("linestyle", "--"),
                linewidth=1.4,
                alpha=0.45,
            )

            ax.text(
                baseline,
                1.01,
                f"{role}: {baseline:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=14,
                color=color,
                alpha=0.85,
                rotation=90,
            )

    # -----------------------------
    # Draw dumbbell plots
    # -----------------------------

    for ax, family in zip(axes, families):
        pdf = plot_base.filter(pl.col(family_col) == family).to_pandas()

        color = family_colors[family]

        for _, row in pdf.iterrows():
            group = row[group_col]
            y = y_positions[group]

            before = row[before_col]
            after = row[after_col]
            increase = row[increase_col]

            x_left = min(before, after)
            x_right = max(before, after)
            x_mid = (before + after) / 2

            # connecting line
            ax.plot(
                [before, after],
                [y, y],
                color=color,
                linewidth=2.0,
                alpha=0.65,
                zorder=1,
            )

            # before point
            ax.scatter(
                before,
                y,
                color="white",
                edgecolor=color,
                linewidth=1.8,
                s=45,
                zorder=3,
            )

            # after point
            ax.scatter(
                after,
                y,
                color=color,
                edgecolor=color,
                linewidth=1.2,
                s=45,
                zorder=4,
            )

            # labels
            ax.text(
                before,
                y + 0.12,
                f"{before:.2f}",
                ha="right",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

            ax.text(
                after,
                y + 0.12,
                f"{after:.2f}",
                ha="left",
                va="bottom",
                fontsize=7,
                color="black",
            )

        ax.set_title(family, fontsize=12, weight="bold", color=color)
        ax.set_xlabel("Accuracy")

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(
            [label_map[g] for g in group_order],
            fontsize=12,  # increase label size
            fontweight="medium",
        )

        ax.grid(axis="x", alpha=0.35)
        ax.grid(axis="y", alpha=0.15)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Group constellation (call cost heuristic)",
        fontsize=13,
        fontweight="bold",
    )
    fig.suptitle(
        "Accuracy Change by Group Constellation and Model Family",
        fontsize=15,
        weight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        1 - 0.01,
        "Rows are ordered by average intra-family rank of final accuracy.",
        ha="center",
        fontsize=10,
    )
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=0,
            label="Accuracy before discussion",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=7,
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=0,
            label="Accuracy after discussion",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            alpha=0.7,
            label="Accuracy improvement",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.45,
            label="Single-model baseline",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    return fig
