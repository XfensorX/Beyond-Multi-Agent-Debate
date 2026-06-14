import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


def baseline_structural_black_hole_plot(knowledge_differences: pl.DataFrame):
    question_overview = (
        knowledge_differences.filter(pl.col("generation_method") == "standard")
        .with_columns(
            (pl.col("model_name") + pl.lit(" | ") + pl.col("combination"))
            .str.split("/")
            .list.last()
            .str.replace("-Reasoning-2512", "")
            .alias("label"),
        )
        .drop(
            [
                "answer_string",
                "final_answer",
                "generation_method",
                "model_name",
                "combination",
                "category",
            ]
        )
        .sort("original_question_id")
    )
    question_overview = (
        question_overview.join(
            question_overview, on="original_question_id", how="inner", suffix="_other"
        )
        .filter(~pl.col("is_correct"))
        .rename(
            {
                "label": "given_that_this_model_is_wrong",
                "label_other": "this_model_is_correct",
            }
        )
        .group_by("given_that_this_model_is_wrong", "this_model_is_correct")
        .agg(
            (pl.col("is_correct_other").sum() / pl.col("is_correct_other").len()).alias(
                "in_that_many_cases"
            )
        )
    )

    parsed_df = question_overview.with_columns(
        [
            pl.col("given_that_this_model_is_wrong")
            .str.split(" | ")
            .list.get(0)
            .alias("wrong_model"),
            pl.col("given_that_this_model_is_wrong")
            .str.split(" | ")
            .list.get(1)
            .alias("wrong_suffix"),
            pl.col("this_model_is_correct")
            .str.split(" | ")
            .list.get(0)
            .alias("correct_model"),
            pl.col("this_model_is_correct")
            .str.split(" | ")
            .list.get(1)
            .alias("correct_suffix"),
        ]
    )
    regex = r"^(.*?)-([\d\.]+)B$"

    df_pd = parsed_df.with_columns(
        [
            pl.col("wrong_model").str.extract(regex, 1).alias("wrong_family"),
            pl.col("wrong_model")
            .str.extract(regex, 2)
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("wrong_params"),
            pl.col("correct_model").str.extract(regex, 1).alias("correct_family"),
            pl.col("correct_model")
            .str.extract(regex, 2)
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("correct_params"),
        ]
    ).to_pandas()

    # Enforce this order by turning the columns into ordered Categoricals
    df_pd["given_that_this_model_is_wrong"] = pd.Categorical(
        df_pd["given_that_this_model_is_wrong"],
        categories=df_pd.sort_values(["wrong_suffix", "wrong_family", "wrong_params"])[
            "given_that_this_model_is_wrong"
        ].unique(),
        ordered=True,
    )
    df_pd["this_model_is_correct"] = pd.Categorical(
        df_pd["this_model_is_correct"],
        categories=df_pd.sort_values(
            ["correct_suffix", "correct_family", "correct_params"], ascending=False
        )["this_model_is_correct"].unique(),
        ordered=True,
    )

    # 3. Pivot the dataframe (Pandas will naturally respect the Categorical order)
    plot_data = df_pd.pivot(
        columns="given_that_this_model_is_wrong",
        index="this_model_is_correct",
        values="in_that_many_cases",
    )

    # 4. Plot the grouped heatmap
    sns.set_style("whitegrid")
    # Note: bumped font_scale slightly. 0.2 can be hard to read depending on figure size.
    sns.set_theme(font="DejaVu Sans", font_scale=0.9)

    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        plot_data, cmap="Blues", annot=True, fmt=".0%", annot_kws={"fontsize": 12}
    )
    ax.set_ylabel("This model is correct in __ part of cases")
    ax.set_xlabel("Given that this model is wrong")

    # 5. Calculate line positions to visually "split" the groups
    # Extract the suffixes from our now perfectly ordered columns and rows
    col_suffixes = [col.split(" | ")[1] for col in plot_data.columns]
    row_suffixes = [row.split(" | ")[1] for row in plot_data.index]

    # Find the exact indices where the suffix switches to a new one
    col_splits = [
        i for i in range(1, len(col_suffixes)) if col_suffixes[i] != col_suffixes[i - 1]
    ]
    row_splits = [
        i for i in range(1, len(row_suffixes)) if row_suffixes[i] != row_suffixes[i - 1]
    ]

    # Draw bold separator lines
    for x in col_splits:
        ax.axvline(x, color="black", lw=2)
    for y in row_splits:
        ax.axhline(y, color="black", lw=2)

    def get_groups(labels):
        """Finds contiguous blocks of the same (Family, Suffix) and their center positions."""
        groups = []
        current_group = ""
        start_idx = 0

        for i, label in enumerate(labels):
            model, suffix = label.split(" | ")
            group_label = f"{suffix}"
            if group_label != current_group:
                if current_group is not None:
                    groups.append((current_group, start_idx, i))
                current_group = group_label
                start_idx = i

        groups.append((current_group, start_idx, len(labels)))
        return groups

    x_groups = get_groups(plot_data.columns)
    y_groups = get_groups(plot_data.index)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # Inherit limits
    x_tick_pos = [(g[1] + g[2]) / 2 for g in x_groups]  # Center of the block
    x_tick_labels = [g[0] for g in x_groups]
    ax_top.set_xticks(x_tick_pos)
    ax_top.set_xticklabels(x_tick_labels, rotation=45, ha="left", fontweight="bold")
    ax_top.grid(False)  # Turn off grid for secondary axis

    # Right Y-Axis
    ax_right = ax.twinx()
    ax_right.set_ylim(
        ax.get_ylim()
    )  # Inherit limits (also inherits heatmap's inverted Y-axis)
    y_tick_pos = [(g[1] + g[2]) / 2 for g in y_groups]
    y_tick_labels = [g[0] for g in y_groups]
    ax_right.set_yticks(y_tick_pos)
    ax_right.set_yticklabels(y_tick_labels, fontweight="bold")
    ax_right.grid(False)

    ax.set_xticklabels(
        [label.split(" | ")[0] for label in plot_data.columns], rotation=45, ha="right"
    )
    ax.set_yticklabels([label.split(" | ")[0] for label in plot_data.index], rotation=0)

    return ax.figure
