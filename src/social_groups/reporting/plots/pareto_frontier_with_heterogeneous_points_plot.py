import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


def pareto_frontier_with_heterogeneous_points_plot(
    no_discussion_baseline, baseline_performance, enhanced_plot_df, homogeneous_mad
):

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="muted",
        font_scale=1.15,
    )

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )

    def pareto_frontier(
        df: pl.DataFrame,
        family_col: str,
        cost_col: str,
        accuracy_col: str,
    ) -> pl.DataFrame:

        return (
            df
            # Keep only the best row for each family/cost pair
            .sort([family_col, cost_col, accuracy_col], descending=[False, False, True])
            .unique(
                subset=[family_col, cost_col],
                keep="first",
                maintain_order=True,
            )
            # Then compute Pareto frontier
            .sort([family_col, cost_col])
            .with_columns(
                prev_best_acc=(pl.col(accuracy_col).cum_max().shift(1).over(family_col))
            )
            .filter(
                pl.col("prev_best_acc").is_null()
                | (pl.col(accuracy_col) > pl.col("prev_best_acc"))
            )
            .drop("prev_best_acc")
        )

    pareto = pareto_frontier(
        no_discussion_baseline,
        family_col="family",
        cost_col="call_cost_heuristic",
        accuracy_col="accuracy_mean",
    )

    pareto_pd = (
        pareto.rename({"family": "model_family"})
        .sort(["model_family", "call_cost_heuristic"])
        .to_pandas()
    )

    baseline_pareto = pareto_frontier(
        baseline_performance,
        family_col="model_family",
        cost_col="call_cost_heuristic",
        accuracy_col="accuracy",
    )

    homogeneous_pareto_pd = (
        pareto_frontier(
            homogeneous_mad,
            family_col="model_family",
            cost_col="call_cost_heuristic",
            accuracy_col="accuracy",
        )
        .sort(["model_family", "call_cost_heuristic"])
        .to_pandas()
    )

    baseline_pareto_pd = baseline_pareto.sort(
        ["model_family", "call_cost_heuristic"]
    ).to_pandas()

    scatter_df = (
        enhanced_plot_df.with_columns(
            outperforms_no_discussion_baseline=(
                pl.col("increase_from_no_discussion_voting_same_calls_baseline")
                .list.eval(pl.element().gt(0.0))
                .list.any()
            ),
            outperforms_best_individual_baseline=(
                pl.col("increase_from_best_individual_baseline")
                .list.eval(pl.element().gt(0.0))
                .list.any()
            ),
        )
        .with_columns(
            outperforms_both_baselines=(
                pl.col("outperforms_no_discussion_baseline")
                & pl.col("outperforms_best_individual_baseline")
            ),
            outperforms_only_no_discussion_baseline=(
                pl.col("outperforms_no_discussion_baseline")
                & ~pl.col("outperforms_best_individual_baseline")
            ),
            label=("(" + pl.col("best_constellation") + ")"),
        )
        .select(
            "model_family",
            "best_constellation",
            "label",
            "best_accuracy",
            "call_cost_heuristic_no_discussion_voting_same_calls_baseline",
            "outperforms_no_discussion_baseline",
            "outperforms_best_individual_baseline",
            "outperforms_both_baselines",
            "outperforms_only_no_discussion_baseline",
        )
        .sort("model_family")
    )

    scatter_pd = scatter_df.to_pandas()

    good_points = scatter_pd[scatter_pd["outperforms_both_baselines"]].copy()
    partial_points = scatter_pd[
        scatter_pd["outperforms_only_no_discussion_baseline"]
    ].copy()
    muted_points = scatter_pd[
        ~scatter_pd["outperforms_both_baselines"]
        & ~scatter_pd["outperforms_only_no_discussion_baseline"]
    ].copy()

    family_order = sorted(
        set(pareto_pd["model_family"])
        | set(baseline_pareto_pd["model_family"])
        | set(homogeneous_pareto_pd["model_family"])
        | set(scatter_pd["model_family"])
    )
    palette = dict(
        zip(
            family_order,
            sns.color_palette("muted", n_colors=len(family_order)),
        )
    )
    n_families = len(family_order)
    fig, axes = plt.subplots(
        1,
        n_families,
        figsize=(5.2 * n_families, 7.5),
        sharex=True,
        sharey=True,
    )

    if n_families == 1:
        axes = [axes]

    for ax, family in zip(axes, family_order):
        family_color = palette[family]

        pareto_f = pareto_pd[pareto_pd["model_family"] == family]
        baseline_f = baseline_pareto_pd[baseline_pareto_pd["model_family"] == family]
        homogeneous_f = homogeneous_pareto_pd[
            homogeneous_pareto_pd["model_family"] == family
        ]

        muted_f = muted_points[muted_points["model_family"] == family]
        partial_f = partial_points[partial_points["model_family"] == family]
        good_f = good_points[good_points["model_family"] == family]

        sns.lineplot(
            data=pareto_f,
            x="call_cost_heuristic",
            y="accuracy_mean",
            color=family_color,
            linewidth=2.5,
            marker="o",
            markersize=6,
            ax=ax,
        )

        sns.lineplot(
            data=baseline_f,
            x="call_cost_heuristic",
            y="accuracy",
            color=family_color,
            linewidth=2.2,
            linestyle=":",
            marker="o",
            markersize=5,
            ax=ax,
        )

        sns.lineplot(
            data=homogeneous_f,
            x="call_cost_heuristic",
            y="accuracy",
            color=family_color,
            linewidth=2.2,
            linestyle="--",
            marker="o",
            markersize=5,
            ax=ax,
        )

        sns.scatterplot(
            data=muted_f,
            x="call_cost_heuristic_no_discussion_voting_same_calls_baseline",
            y="best_accuracy",
            color=family_color,
            s=45,
            alpha=0.4,
            edgecolor="none",
            ax=ax,
            legend=False,
        )

        sns.scatterplot(
            data=partial_f,
            x="call_cost_heuristic_no_discussion_voting_same_calls_baseline",
            y="best_accuracy",
            color=family_color,
            s=65,
            alpha=0.7,
            edgecolor="gray",
            linewidth=0.4,
            ax=ax,
            legend=False,
        )

        sns.scatterplot(
            data=good_f,
            x="call_cost_heuristic_no_discussion_voting_same_calls_baseline",
            y="best_accuracy",
            color=family_color,
            s=90,
            edgecolor="black",
            linewidth=0.6,
            ax=ax,
            legend=False,
        )

        texts = []

        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        dx = -0.006 * x_range
        dy = 0.006 * y_range

        for _, row in good_f.iterrows():
            texts.append(
                ax.text(
                    row["call_cost_heuristic_no_discussion_voting_same_calls_baseline"]
                    + dx,
                    row["best_accuracy"] + dy,
                    row["label"],
                    fontsize=8.5,
                    weight="semibold",
                    ha="right",
                    va="bottom",
                    color="black",
                )
            )

        for _, row in partial_f.iterrows():
            texts.append(
                ax.text(
                    row["call_cost_heuristic_no_discussion_voting_same_calls_baseline"]
                    + dx,
                    row["best_accuracy"] + dy,
                    row["label"],
                    fontsize=7.2,
                    weight="normal",
                    ha="right",
                    va="bottom",
                    color="gray",
                    alpha=0.75,
                )
            )

        if HAS_ADJUST_TEXT:
            adjust_text(
                texts,
                ax=ax,
                only_move={"points": "xy", "text": "xy"},
                expand_points=(1.3, 1.6),
                expand_text=(1.15, 1.35),
                force_points=(0.25, 0.45),
                force_text=(0.35, 0.7),
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.5,
                    alpha=0.45,
                    color="gray",
                ),
            )

        ax.set_title(f"Model family: {family}")
        ax.set_xlabel("Call cost heuristic (2 x params per call)")
        ax.grid(True, alpha=0.25)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Accuracy")

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_xlim(0, 250)
        ax.set_ylim(0.3, 0.8)

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.5,
            linestyle="-",
            label="No Discussion Baseline Pareto Frontier",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.2,
            linestyle=":",
            label="Single Model Baseline Pareto Frontier",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.2,
            linestyle="--",
            label="Homogeneous MAD Pareto Frontier",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="gray",
            markersize=8,
            linestyle="None",
            label="Outperforms both baselines",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="gray",
            markersize=6,
            alpha=0.55,
            linestyle="None",
            label="Outperforms no-discussion baseline only",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markerfacecolor="gray",
            markersize=5,
            alpha=0.28,
            linestyle="None",
            label="Other cases",
        ),
    ]

    fig.legend(
        handles=style_handles,
        title="Series",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        frameon=True,
    )

    fig.suptitle(
        "Accuracy - Cost Tradeoff for Size-Heterogeneous MAD",
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig
