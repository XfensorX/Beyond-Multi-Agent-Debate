import polars as pl


def baseline_few_shot_comparison_plot(pretty_main: pl.DataFrame):
    improvment_few_shot = (
        pretty_main.filter(pl.col("Method") == "Chat Completion")
        .group_by("Family", "Model", "Reasoning")
        .agg(
            pl.when((pl.col("Few Shot Prompting") == "Yes"))
            .then(pl.col("Accuracy (Advanced)") - pl.col("Accuracy (Naive)"))
            .alias("Naive -> Advanced (5-Shot)")
            .drop_nulls()
            .item(),
            pl.when((pl.col("Few Shot Prompting") == "No"))
            .then(pl.col("Accuracy (Advanced)") - pl.col("Accuracy (Naive)"))
            .alias("Naive -> Advanced (0-Shot)")
            .drop_nulls()
            .item(),
            pl.when(pl.col("Few Shot Prompting") == "Yes")
            .then(pl.col("Accuracy (Naive)"))
            .alias("temp_5shot_naive")
            .drop_nulls()
            .first(),
            pl.when(pl.col("Few Shot Prompting") == "No")
            .then(pl.col("Accuracy (Naive)"))
            .alias("temp_0shot_naive")
            .drop_nulls()
            .first(),
            pl.when(pl.col("Few Shot Prompting") == "Yes")
            .then(pl.col("Accuracy (Advanced)"))
            .alias("temp_5shot_adv")
            .drop_nulls()
            .first(),
            pl.when(pl.col("Few Shot Prompting") == "No")
            .then(pl.col("Accuracy (Advanced)"))
            .alias("temp_0shot_adv")
            .drop_nulls()
            .first(),
        )
        .with_columns(
            (pl.col("temp_5shot_naive") - pl.col("temp_0shot_naive")).alias(
                "0-Shot -> 5-Shot (Naive)"
            ),
            (pl.col("temp_5shot_adv") - pl.col("temp_0shot_adv")).alias(
                "0-Shot -> 5-Shot (Advanced)"
            ),
        )
        .drop(
            ["temp_5shot_naive", "temp_0shot_naive", "temp_5shot_adv", "temp_0shot_adv"]
        )
        .sort("Family", descending=True)
        .with_columns(
            sorting=pl.col("Model")
            .str.split("-")
            .list.last()
            .str.replace("B", "")
            .cast(float)
        )
        .sort("Family", "sorting", descending=True)
        .drop("sorting")
    )

    df_plot = improvment_few_shot.to_pandas().copy()

    # Create a nice label for the y-axis
    df_plot["Model Label"] = df_plot["Model"]

    # Melt into long format
    df_long = df_plot.melt(
        id_vars=["Family", "Model", "Model Label", "Reasoning"],
        value_vars=[
            "Naive -> Advanced (5-Shot)",
            "Naive -> Advanced (0-Shot)",
            "0-Shot -> 5-Shot (Naive)",
            "0-Shot -> 5-Shot (Advanced)",
        ],
        var_name="Metric",
        value_name="Delta",
    )

    # Create the two main categories for columns
    df_long["Improvement Category"] = df_long["Metric"].apply(
        lambda x: (
            "Naive → Advanced Parsing"
            if "Naive -> Advanced" in x
            else "0-Shot → 5-Shot"
        )
    )

    # Create the sub-type (what appears inside each column)
    df_long["Shot Type"] = df_long["Metric"].replace(
        {
            "Naive -> Advanced (5-Shot)": "5-Shot",
            "Naive -> Advanced (0-Shot)": "0-Shot",
            "0-Shot -> 5-Shot (Naive)": "Naive",
            "0-Shot -> 5-Shot (Advanced)": "Advanced",
        }
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    # NeurIPS-style setup: clean, high-quality, sans-serif
    sns.set_theme(style="white", context="paper")  # "white" for clean paper look

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # or "Arial" / "Helvetica" if available
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.8,  # bar edge
            "figure.dpi": 300,  # high-res for saving
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    # Main plot with better proportions
    g = sns.catplot(
        data=df_long,
        kind="bar",
        x="Delta",
        y="Model Label",
        hue="Shot Type",
        col="Improvement Category",
        row="Reasoning",
        orient="h",
        palette="muted",  # or "tab10", "Set2" for more contrast
        height=7.0,  # taller per facet for bigger bars
        aspect=1.2,  # wider aspect → bigger bars, less squished
        width=1.0,  # thicker bars (0.6-0.85 works well)
        dodge=True,  # ensure hue bars don't overlap
        legend=True,
        sharex=True,
        sharey=False,  # y can differ per row if needed
        margin_titles=True,
        gap=0.45,
        legend_out=True,
    )

    g.legend.set_loc("lower center")
    g.legend.set_bbox_to_anchor([0.85, 0.87])

    # Suptitle (NeurIPS-style: bold, centered, slightly larger)
    g.figure.suptitle(
        "Performance Gains from Prompting Strategies on Standard Generation (Percentage Points)",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )

    # Axis labels
    g.set_axis_labels("Δ Accuracy (percentage points)", "")
    g.set_titles(
        row_template="Reasoning: {row_name}",
        col_template="{col_name}",
        size=11,
        weight="semibold",
    )

    # Improve bar labels: larger, better positioned, avoid overlap
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f",  # one decimal is usually enough
                padding=4,
                fontsize=9,
                label_type="edge",  # or "center" if bars are wide enough
                color="black",
            )

    # Clean axes appearance (NeurIPS-like)
    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="x")
        ax.set_axisbelow(True)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.9, alpha=0.75)
        sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)
        ax.tick_params(axis="y", length=0, pad=6)

    # for ax in g.axes.flat:
    #     ax.legend(g._legend_data, title="", loc="upper right") # Customize location as needed

    # Final figure size (adjust based on your number of rows/cols)
    # Example: ~2 rows × several cols → wide landscape
    g.figure.set_size_inches(w=16, h=9)  # tweak: e.g., (16, 12) or (20, 11)

    return g
