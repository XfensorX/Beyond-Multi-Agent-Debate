import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D


def homogeneous_mad_cost_tradeoff_paretofront_plot(df: pl.DataFrame, log_x=True):
    SIZES_MIN_MAX = (40, 140)

    METHODS = {
        "Multi Agent Debate": {
            "accuracy": "accuracy",
            "suffix": "",
        },
        "Best individual": {
            "accuracy": "accuracy_best_individual_baseline",
            "suffix": "_best_individual_baseline",
        },
        "Same-calls voting": {
            "accuracy": "accuracy_no_discussion_voting_same_calls_baseline",
            "suffix": "_no_discussion_voting_same_calls_baseline",
        },
    }

    COSTS = {
        "Input tokens": "used_input_tokens",
        "Output tokens": "used_output_tokens",
        "Total tokens": "used_total_tokens",
        "Call Cost Heuristic": "call_cost_heuristic",
    }

    def existing_columns(df, *cols):
        return all(col in df.columns for col in cols)

    def make_long_df(df: pl.DataFrame) -> pl.DataFrame:
        rows = []

        for method_name, method in METHODS.items():
            acc_col = method["accuracy"]

            if acc_col not in df.columns:
                continue

            for cost_name, base_cost_col in COSTS.items():
                cost_col = base_cost_col + method["suffix"]

                if cost_col not in df.columns:
                    continue

                rows.append(
                    df.select(
                        pl.col("model_family"),
                        pl.col("group_constellation"),
                        pl.lit(method_name).alias("method"),
                        pl.lit(cost_name).alias("cost_metric"),
                        pl.col(acc_col).alias("accuracy"),
                        pl.col(cost_col).alias("cost"),
                        pl.col("parameters"),
                    )
                )

        if not rows:
            raise ValueError("No matching method/cost columns found.")

        return pl.concat(rows).filter(
            pl.col("accuracy").is_not_null()
            & pl.col("cost").is_not_null()
            & (pl.col("cost") > 0)
        )

    def pareto_frontier(data):
        """
        Lower cost is better, higher accuracy is better.
        """
        data = data.sort_values(["cost", "accuracy"], ascending=[True, False])

        best_accuracy_so_far = -np.inf
        frontier_rows = []

        for _, row in data.iterrows():
            if row["accuracy"] > best_accuracy_so_far:
                frontier_rows.append(row)
                best_accuracy_so_far = row["accuracy"]

        return data.__class__(frontier_rows)

    data = make_long_df(
        df.with_columns(pl.selectors.numeric().cast(pl.Float64))
    ).to_pandas()

    cost_metrics = list(COSTS.keys())
    cost_metrics = [c for c in cost_metrics if c in data["cost_metric"].unique()]

    methods = list(METHODS.keys())
    methods = [m for m in methods if m in data["method"].unique()]

    families = sorted(data["model_family"].dropna().unique())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = dict(zip(families, markers))

    palette = dict(zip(methods, sns.color_palette("colorblind", len(methods))))

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    ncols = 2
    nrows = int(np.ceil(len(cost_metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    if len(cost_metrics) == 1:
        axes = [axes]

    for ax, cost_metric in zip(axes, cost_metrics):
        panel = data[data["cost_metric"] == cost_metric]

        sns.scatterplot(
            data=panel,
            x="cost",
            y="accuracy",
            hue="method",
            hue_order=methods,
            palette=palette,
            style="model_family",
            style_order=families,
            markers=marker_map,
            size="parameters",
            sizes=SIZES_MIN_MAX,
            s=65,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            legend=False,
            ax=ax,
        )

        frontier = pareto_frontier(panel)

        ax.plot(
            frontier["cost"],
            frontier["accuracy"],
            color="black",
            linewidth=1,
            marker="o",
            markersize=2.5,
            label="Pareto frontier",
        )

        if log_x:
            ax.set_xscale("log")

        ax.set_title(cost_metric)
        ax.set_xlabel("")
        ax.grid(alpha=0.35)

    for ax in axes[len(cost_metrics) :]:
        ax.set_visible(False)

    axes[0].set_ylabel("Accuracy")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=method,
            markerfacecolor=palette[method],
            markeredgecolor="white",
            markersize=7,
        )
        for method in methods
    ]

    legend_items.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            label="Pareto frontier",
            linewidth=2,
            markersize=5,
        )
    )

    family_legend_items = [
        Line2D(
            [0],
            [0],
            marker=marker_map[family],
            linestyle="",
            label=family,
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=7,
        )
        for family in families
    ]

    method_legend = fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=len(legend_items),
        frameon=True,
        bbox_to_anchor=(0.4, 1.10),
    )

    family_legend = fig.legend(
        handles=family_legend_items,
        loc="upper center",
        ncol=min(len(family_legend_items), 5),
        frameon=True,
        bbox_to_anchor=(0.23, 1.06),
    )

    size_handles = [
        plt.scatter(
            [],
            [],
            s=np.interp(
                v,
                [data["parameters"].min(), data["parameters"].max()],
                list(SIZES_MIN_MAX),
            ),
            color="gray",
            alpha=0.85,
            label=f"{v:.1f}B",
        )
        for v in [0.6, 4, 9, 14]
    ]

    size_legend = fig.legend(
        handles=size_handles,
        title=" Individual Model \nParameters Count",
        loc="upper center",
        bbox_to_anchor=(0.9, 1.16),
        frameon=True,
        fontsize="small",
    )

    fig.suptitle(
        "Accuracy–Cost Pareto Tradeoff (Homogeneous MAD)", fontweight="bold", y=1.15
    )

    return fig
