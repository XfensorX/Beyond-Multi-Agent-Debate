import re
from re import Match

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D


def _load_seaborn_settings():

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"


def baseline_method_comparison_plot(main_result: pl.DataFrame):
    _load_seaborn_settings()

    plotting_df = (
        main_result.with_columns(
            pl.col("model_name")
            .str.extract(r"/(Qwen[0-9.]*|Ministral-3)")
            .alias("Family")
        )
        .fill_null(False)
        .with_columns(
            pl.concat_str(
                [
                    pl.when(pl.col("use_thinking"))
                    .then(pl.lit("R"))
                    .otherwise(pl.lit("x")),
                    pl.when(pl.col("with_few_shot_prompting"))
                    .then(pl.lit("F"))
                    .otherwise(pl.lit("x")),
                ]
            ).alias("combination")
        )
    )

    combo_order = plotting_df.get_column("combination").unique().sort().to_list()

    plotting_df = plotting_df.to_pandas()
    plotting_df["combination"] = pd.Categorical(
        plotting_df["combination"], categories=combo_order, ordered=True
    )

    plotting_df_long = (
        plotting_df.melt(
            id_vars=["generation_method", "combination", "model_name", "Family"],
            value_vars=[
                "Accuracy (Naive)",
                "Accuracy (Advanced)",  # add more if needed
            ],
            var_name="metric",
            value_name="value",
        )
        .replace(
            {
                "Accuracy (Naive)": "Accuracy (Naive Parsing)",
                "Accuracy (Advanced)": "Accuracy (Advanced RegEx)",
            }
        )
        .sort_values(["Family", "combination", "model_name", "generation_method"])
    )

    families = plotting_df_long["Family"].unique()
    model_names = plotting_df_long["model_name"].unique()
    palette = {
        m: c
        for m, c in zip(
            model_names, sns.color_palette("muted", n_colors=len(model_names))
        )
    }

    g = sns.relplot(
        data=plotting_df_long,
        x="combination",
        y="value",
        hue="model_name",
        style="generation_method",
        markers={"tool": "^", "standard": "s", "structured-output": "o"},
        dashes={"standard": (None, None), "tool": (4, 4), "structured-output": (1, 3)},
        col="Family",
        row="metric",
        kind="line",
        height=5,
        aspect=2.0,
        palette=palette,
        facet_kws={"margin_titles": True},
        legend=False,
    )

    g.set_axis_labels("", "%")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    g.figure.suptitle(
        "Accuracy by Model Family, Generation Method and Parsing Option (MMLUPro 1000-Subset)",
        y=1.05,
        fontsize=18,
        fontweight="bold",
    )

    style_handles = [
        Line2D(
            [0], [0], color="black", lw=2, linestyle="-", marker="s", label="Standard"
        ),
        Line2D([0], [0], color="black", lw=2, linestyle="--", marker="^", label="Tool"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2,
            linestyle=":",
            marker="o",
            label="Structured Output",
        ),
    ]

    g.figure.legend(
        handles=style_handles,
        title="Generation Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    g.figure.supxlabel(
        "Settings Combination (R = Reasoning | F = Few-Shot)",
        y=0.16,  # adjust this value if needed (lower = more space from bottom)
        fontsize=18,  # optional: adjust size
    )

    for ax, family in zip(g.axes[-1], families):
        handles = [
            Line2D(
                [0],
                [0],
                color=palette[m],
                lw=3,
                marker="o",
                label=m.split("/")[-1].replace("Reasoning", "__"),
            )
            for m in sorted(
                model_names,
                key=lambda x: (
                    -float((re.search(r"(\d+\.?\d*)B", x) or Match()).group(1))
                ),
            )
            if f"/{family}-" in m
        ]

        ax.legend(
            handles=handles,
            # title=f"{family} - Family",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.5),
            ncol=1,
            frameon=True,
        )

    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    for i, ax_row in enumerate(g.axes):
        for ax in ax_row:
            ax.set_facecolor("#f9f9f9")

    # Subtle grid + thin lines
    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.9)
        for line in ax.lines:
            line.set_linewidth(1.5)

    alpha_map = {
        "standard": 1.0,
        "tool": 0.8,
        "structured-output": 0.6,
    }

    linewidth_map = {
        "standard": 1.5,
        "tool": 1.2,
        "structured-output": 1.2,
    }

    desaturate_map = {
        "standard": 1.0,
        "tool": 0.9,
        "structured-output": 0.7,
    }

    def get_method_from_line(line):
        linestyle = line.get_linestyle()

        # Matplotlib normalizes these internally
        if linestyle == "-" or linestyle == "solid":
            return "standard"
        elif linestyle == "--":
            return "tool"  # (4,4) often becomes '--'
        elif linestyle == ":":
            return "structured-output"  # (1,3) often becomes ':'
        else:
            return "standard"  # fallback

    def desaturate(color, amount=0.7):
        c = matplotlib.colors.to_rgb(color)
        return tuple(1 - amount * (1 - x) for x in c)

    for ax in g.axes.flat:
        for line in ax.lines:
            method = get_method_from_line(line)
            line.set_alpha(alpha_map[method])
            line.set_linewidth(linewidth_map[method])
            line.set_color(desaturate(line.get_color(), desaturate_map[method]))
            line.set_markersize(8)

    g.figure.set_size_inches(20, 15)
    g.tight_layout()

    return g
