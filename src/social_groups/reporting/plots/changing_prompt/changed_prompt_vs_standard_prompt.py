import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


def changed_prompt_vs_standard_prompt(combined_results_with_wd):

    # -----------------------------
    # Configuration
    # -----------------------------

    metrics = [
        "accuracy",
        "accuracy_before",
        "amount_parsing_errors_before",
        "accuracy_increase",
        "wd_bias_False",
    ]

    metric_titles = {
        "accuracy": "Δ Accuracy",
        "accuracy_before": "Δ Accuracy Before Debate",
        "amount_parsing_errors_before": "Δ Parsing Errors Before Debate",
        "accuracy_increase": "Δ Accuracy Increase",
        "wd_bias_False": "Δ WD (No Bias)",
    }

    index_col = "group_constellation"
    column_col = "model_family"

    # -----------------------------
    # Ordering heuristic
    # H-heavy at top, M-middle, L-heavy bottom
    # -----------------------------

    def constellation_sort_key(s: str):
        s = str(s)

        n_h = s.count("H")
        n_m = s.count("M")
        n_l = s.count("L")

        # Higher score = should appear higher
        strength_score = (2 * n_h) + (1 * n_m) - (2 * n_l)

        # Tie-breakers:
        # 1. More H higher
        # 2. More M before more L
        # 3. Fewer L higher
        # 4. Shorter constellations first
        # 5. Alphabetical fallback
        return (
            -strength_score,
            -n_h,
            -n_m,
            n_l,
            len(s),
            s,
        )

    # -----------------------------
    # Compute deltas
    # -----------------------------

    delta_df = combined_results_with_wd.with_columns(
        [(pl.col(f"{m}_new_prompt") - pl.col(m)).alias(f"{m}_delta") for m in metrics]
    )

    # Keep only rows that have at least one usable comparison
    delta_cols = [f"{m}_delta" for m in metrics]

    delta_df = delta_df.filter(
        pl.any_horizontal(
            [pl.col(c).is_not_null() & ~pl.col(c).is_nan() for c in delta_cols]
        )
    )

    row_order = sorted(
        delta_df.get_column(index_col).unique().to_list(),
        key=constellation_sort_key,
    )

    # -----------------------------
    # Plot styling
    # -----------------------------

    sns.set_theme(
        style="white",
        context="paper",
        palette="muted",
        font_scale=0.95,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(13.5, 6),
        sharey=True,
    )

    # -----------------------------
    # Draw heatmaps
    # -----------------------------

    for ax, metric in zip(axes, metrics):
        value_col = f"{metric}_delta"

        plot_df = (
            delta_df.select(index_col, column_col, value_col)
            .drop_nulls([value_col])
            .filter(~pl.col(value_col).is_nan())
            .group_by(index_col, column_col)
            .agg(pl.mean(value_col).alias(value_col))
            .to_pandas()
            .pivot(
                index=index_col,
                columns=column_col,
                values=value_col,
            )
            .reindex(row_order)
            .dropna(how="all")
        )

        max_abs = plot_df.abs().max().max()

        if max_abs == 0 or max_abs != max_abs:
            max_abs = 1e-9

        norm = TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs,
        )

        sns.heatmap(
            plot_df,
            ax=ax,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            norm=norm,
            linewidths=0.35,
            linecolor="white",
            cbar=True,
            cbar_kws={
                "label": metric_titles[metric],
                "shrink": 0.75,
            },
            square=False,
        )

        ax.set_title(
            metric_titles[metric],
            fontsize=11,
            weight="bold",
            pad=10,
        )

        ax.set_xlabel("Model family")
        ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.tick_params(
            axis="y",
            labelsize=8,
            labelrotation=0,
        )

    axes[0].set_ylabel("Group constellation")

    fig.suptitle(
        "Effect of New Prompt Relative to Standard Prompt",
        fontsize=14,
        weight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        -0.01,
        "Cells show the changes from switching MAD from standard to the changed prompts. Positive values indicate an increase under the new prompt.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    return fig
