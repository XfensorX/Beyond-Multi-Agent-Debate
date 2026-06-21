import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


def permutation_effect_within_each_model_family(only_changed_prompt):

    metrics = [
        "accuracy",
        "accuracy_increase",
        "wd_bias_False",
        "accuracy_before",
    ]

    metric_titles = {
        "accuracy": "Accuracy",
        "accuracy_increase": "Accuracy increase",
        "wd_bias_False": "WD bias False",
        "accuracy_before": "Accuracy before",
    }

    # -----------------------------
    # Helpers
    # -----------------------------

    def signature_expr(col="group_constellation"):
        return pl.col(col).str.split("").list.sort().list.join("")

    def is_homogeneous_constellation(s: str) -> bool:
        s = str(s)
        return len(set(s)) == 1

    def constellation_sort_key(s: str):
        s = str(s)

        n_h = s.count("H")
        n_m = s.count("M")
        n_l = s.count("L")

        strength_score = (2 * n_h) + n_m - (2 * n_l)

        return (
            -strength_score,  # H-heavy higher
            -n_h,
            -n_m,
            n_l,
            len(s),
            s,
        )

    # -----------------------------
    # Aggregate and compute effects
    # -----------------------------

    base_df = (
        only_changed_prompt.with_columns(signature_expr().alias("composition"))
        .group_by("model_family", "composition", "group_constellation")
        .agg([pl.col(m).item().alias(m) for m in metrics])
    )

    # IMPORTANT:
    # Mean is computed within model_family and composition.
    # There is no averaging across model families.
    mean_df = base_df.group_by("model_family", "composition").agg(
        [pl.mean(m).alias(f"{m}_composition_mean") for m in metrics]
    )

    effect_df = base_df.join(mean_df, on=["model_family", "composition"]).with_columns(
        [
            (pl.col(m) - pl.col(f"{m}_composition_mean")).alias(f"{m}_effect")
            for m in metrics
        ]
    )

    # -----------------------------
    # Row order:
    # heterogeneous groups first,
    # homogeneous groups collected at bottom
    # -----------------------------

    def composition_key(s: str):
        return "".join(sorted(str(s)))

    def within_permutation_key(s: str):
        mapping = str.maketrans({"L": "3", "M": "2", "H": "1"})
        return int(str(s)[::-1].translate(mapping))

    all_groups = effect_df.get_column("group_constellation").unique().to_list()

    heterogeneous_groups = [
        g for g in all_groups if not is_homogeneous_constellation(g)
    ]
    homogeneous_groups = [g for g in all_groups if is_homogeneous_constellation(g)]

    groups_by_composition = {}
    for g in heterogeneous_groups:
        comp = composition_key(g)
        groups_by_composition.setdefault(comp, []).append(g)

    composition_order = sorted(
        groups_by_composition.keys(),
        key=constellation_sort_key,
    )

    row_order = []
    separator_positions = []

    for comp in composition_order:
        ordered_group = sorted(
            groups_by_composition[comp],
            key=within_permutation_key,
        )
        row_order.extend(ordered_group)
        separator_positions.append(len(row_order))

    homogeneous_start_idx = len(row_order)

    row_order.extend(sorted(homogeneous_groups, key=constellation_sort_key))

    separator_positions = separator_positions[:-1]

    if homogeneous_groups:
        separator_positions.append(homogeneous_start_idx)

    # -----------------------------
    # Plot
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
        figsize=(13.5, 10),
        sharey=True,
    )

    global_max_abs = max(max(effect_df[f"{m}_effect"].max(), 1e-9) for m in metrics)

    for ax, metric in zip(axes, metrics):
        value_col = f"{metric}_effect"

        plot_df = (
            effect_df.select("group_constellation", "model_family", value_col)
            .group_by("group_constellation", "model_family")
            .agg(pl.mean(value_col).alias(value_col))
            .to_pandas()
            .pivot(
                index="group_constellation",
                columns="model_family",
                values=value_col,
            )
            .reindex(row_order)
        )

        max_abs = plot_df.abs().max().max()

        norm = TwoSlopeNorm(
            vmin=-global_max_abs if not metric == "wd_bias_False" else -max_abs,
            vcenter=0,
            vmax=global_max_abs if not metric == "wd_bias_False" else max_abs,
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
                "label": f"{metric_titles[metric]} − composition mean",
                "shrink": 0.75,
            },
        )

        for pos in separator_positions:
            ax.axhline(
                pos,
                color="black",
                linewidth=1.4,
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
        ax.tick_params(axis="y", labelsize=8)

    axes[0].set_ylabel("Group constellation")

    fig.suptitle(
        "Permutation Effect Within Each Model Family",
        fontsize=14,
        weight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        1 - 0.01,
        "Cells show <metric − mean(metric)> within the same model family and letter composition. "
        "Homogeneous constellations are grouped at the bottom.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    return fig
