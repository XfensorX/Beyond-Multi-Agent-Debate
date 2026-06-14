import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def improvement_per_family_constellation_heatmap_plot(
    df: pl.DataFrame, improvement_cols: list[str], title: str
):
    sns.set_theme(font_scale=1.6)
    CMAP = LinearSegmentedColormap.from_list(
        "red_white_green",
        [
            (0.0, "#b30000"),  # dark red
            (0.25, "#ff6666"),  # light red
            (0.50, "#ffffff"),  # pure white at zero
            (0.75, "#66cc66"),  # light green
            (1.0, "#006400"),  # dark green
        ],
    )
    pdf = df.to_pandas()

    pivots = [
        pdf.pivot(
            index="model_family",
            columns="group_constellation",
            values=col,
        )
        for col in improvement_cols
    ]

    family_order = (
        pdf.groupby("model_family")[improvement_cols[0]]
        .median()
        .sort_values(ascending=False)
        .index
    )

    pivots = [p.loc[family_order] for p in pivots]

    def constellation_sort_key(s: str):
        counts = {
            "L": s.count("L"),
            "M": s.count("M"),
            "H": s.count("H"),
        }
        return (
            counts["L"] + 2 * counts["M"] + 3 * counts["H"],
            len(set(s)) > 1,
            -counts["L"],
            -counts["M"],
            -counts["H"],
            len(s),
            s,
        )

    constellation_order = sorted(
        pdf["group_constellation"].unique(),
        key=constellation_sort_key,
    )
    pivots = [p[constellation_order] for p in pivots]

    max_val = max(((pdf[c]).max() for c in improvement_cols))
    min_val = min(((pdf[c]).min() for c in improvement_cols))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(6, int(len(family_order) * 0.4))),
        constrained_layout=True,
    )

    for ax_id, p in enumerate(pivots):
        sns.heatmap(
            p,
            annot=p.map(lambda x: f"{x * 100:.1f}"),
            fmt="",
            cmap=CMAP,
            center=0,
            vmin=min_val,
            vmax=max_val,
            linewidths=0.5,
            ax=axes[ax_id],
            cbar_kws={"label": "Accuracy Improvement"}
            if ax_id == len(pivots) - 1
            else None,
            # annot_kws={"fontsize": 10},
        )

        axes[ax_id].set_title(
            f"Improvement vs {improvement_cols[ax_id]} (in p.p.)",
        )
        axes[ax_id].set_xlabel("Group Constellation")
        if ax_id == 0:
            axes[ax_id].set_ylabel("Model Family")
        else:
            axes[ax_id].set_ylabel("")

    fig.suptitle(title, y=1.1)

    return fig
