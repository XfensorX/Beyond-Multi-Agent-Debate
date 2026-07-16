import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def baseline_few_shot_comparison_plot(pretty_main: pl.DataFrame):
    # === 1. Focus ONLY on 0-Shot → 5-Shot gains (right side of original plot) ===
    improvment_few_shot = (
        pretty_main.filter(pl.col("Method") == "Chat Completion")
        .group_by("Family", "Model", "Reasoning")
        .agg(
            pl.when(pl.col("Few Shot Prompting") == "Yes")
            .then(pl.col("Accuracy (Naive)"))
            .alias("temp_5shot_naive")
            .drop_nulls()
            .item(),
            pl.when(pl.col("Few Shot Prompting") == "No")
            .then(pl.col("Accuracy (Naive)"))
            .alias("temp_0shot_naive")
            .drop_nulls()
            .item(),
            pl.when(pl.col("Few Shot Prompting") == "Yes")
            .then(pl.col("Accuracy (Advanced)"))
            .alias("temp_5shot_adv")
            .drop_nulls()
            .item(),
            pl.when(pl.col("Few Shot Prompting") == "No")
            .then(pl.col("Accuracy (Advanced)"))
            .alias("temp_0shot_adv")
            .drop_nulls()
            .item(),
        )
        .with_columns(
            (pl.col("temp_5shot_naive") - pl.col("temp_0shot_naive")).alias(
                "0-Shot → 5-Shot (Naive)"
            ),
            (pl.col("temp_5shot_adv") - pl.col("temp_0shot_adv")).alias(
                "0-Shot → 5-Shot (Advanced)"
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

    # Nice model label
    df_plot["Model Label"] = df_plot["Model"]

    # Melt to long format — now only the two relevant columns
    df_long = df_plot.melt(
        id_vars=["Family", "Model", "Model Label", "Reasoning"],
        value_vars=[
            "0-Shot → 5-Shot (Naive)",
            "0-Shot → 5-Shot (Advanced)",
        ],
        var_name="Parsing Type",
        value_name="Delta",
    )

    # Clean up labels
    df_long["Parsing Type"] = df_long["Parsing Type"].replace(
        {
            "0-Shot → 5-Shot (Naive)": "Naive",
            "0-Shot → 5-Shot (Advanced)": "Advanced",
        }
    )

    # ====================== PLOTTING ======================
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    # One figure with two subplots side-by-side (Naive vs Advanced)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    palette = sns.color_palette("muted", n_colors=2)  # one color per Reasoning level

    for ax, parsing_type in zip(axes, ["Naive", "Advanced"]):
        data = df_long[df_long["Parsing Type"] == parsing_type]

        # Vertical bars (bottom to top)
        sns.barplot(
            data=data,
            x="Model Label",
            y="Delta",
            hue="Reasoning",
            palette=palette,
            ax=ax,
            dodge=True,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_title(f"{parsing_type} Parsing", fontsize=13, pad=15)
        ax.set_xlabel("")
        ax.set_ylabel(
            "Δ Accuracy (percentage points) from without to with few-shot prompting"
            if ax == axes[0]
            else ""
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0)
        ax.tick_params(axis="x", rotation=45, length=0, pad=8)
        sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)

        # Bar labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9, color="black")

    # Overall title
    fig.suptitle(
        "Performance Gains from using few-shot prompting",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Legend at the bottom center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Reasoning",
        title_fontsize=11,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )

    # Remove individual subplot legends
    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle + legend
    return fig
