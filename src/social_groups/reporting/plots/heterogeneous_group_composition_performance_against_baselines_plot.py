import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def heterogeneous_group_composition_performance_against_baselines_plot(plot_df):
    # -----------------------------
    # 2. Convert to pandas for plotting
    # -----------------------------

    pdf = plot_df.to_pandas()

    model_families = sorted(pdf["model_family"].unique())
    n_families = len(model_families)

    # Helper: accuracy is now a list; use its mean only for ordering/comparison
    def as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    pdf["accuracy_mean"] = pdf["accuracy"].apply(lambda x: np.mean(as_list(x)))

    # -----------------------------
    # 3. Global group ordering
    # -----------------------------

    group_order = (
        pdf.groupby("group_composition")["accuracy_mean"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # -----------------------------
    # 4. Colors and markers
    # -----------------------------

    main_color = "black"
    best_individual_color = "#E69F00"  # orange
    voting_color = "#0072B2"  # darker sky-blue

    main_alpha = 0.55
    baseline_alpha = 0.85
    line_alpha = 0.55

    marker_map = {
        "accuracy_best_individual_baseline": "^",  # triangle
        "accuracy_no_discussion_voting_same_calls_baseline": "s",
        "accuracy": "o",  # circle
    }

    # -----------------------------
    # 5. Create figure
    # -----------------------------

    fig_height = 0.45 * len(group_order) + 2.5

    fig, axes = plt.subplots(
        1,
        n_families,
        figsize=(5.2 * n_families, fig_height),
        sharex=True,
        sharey=True,
    )

    if n_families == 1:
        axes = [axes]

    # -----------------------------
    # 6. Plot each model family
    # -----------------------------

    for ax, family in zip(axes, model_families):
        sub = (
            pdf[pdf["model_family"] == family]
            .set_index("group_composition")
            .reindex(group_order)
            .reset_index()
        )

        y_positions = np.arange(len(group_order))

        # very light horizontal guide lines
        for y in y_positions:
            ax.axhline(y, color="0.92", linewidth=0.8, zorder=0)

        for y, (_, row) in zip(y_positions, sub.iterrows()):
            required_cols = [
                "accuracy",
                "accuracy_best_individual_baseline",
                "accuracy_no_discussion_voting_same_calls_baseline",
            ]

            if row[required_cols].isna().any():
                continue

            main_acc_values = as_list(row["accuracy"])
            main_acc_mean = np.mean(main_acc_values)

            baselines = [
                (
                    row["accuracy_best_individual_baseline"],
                    "accuracy_best_individual_baseline",
                    best_individual_color,
                ),
                (
                    row["accuracy_no_discussion_voting_same_calls_baseline"],
                    "accuracy_no_discussion_voting_same_calls_baseline",
                    voting_color,
                ),
            ]

            for baseline_acc, baseline_col, color in baselines:
                # Connecting line from baseline to mean of main-method runs
                ax.plot(
                    [baseline_acc, main_acc_mean],
                    [y, y],
                    color=color,
                    linewidth=1.3,
                    alpha=line_alpha,
                    zorder=1,
                )

                # Baseline point
                ax.scatter(
                    baseline_acc,
                    y,
                    marker=marker_map[baseline_col],
                    s=66,
                    color=color,
                    alpha=baseline_alpha,
                    zorder=3,
                )

            # Main method: draw every accuracy value at the same height
            ax.scatter(
                main_acc_values,
                [y] * len(main_acc_values),
                marker=marker_map["accuracy"],
                s=40,
                color=main_color,
                alpha=main_alpha,
                zorder=4,
            )

        ax.set_title(family)
        ax.set_xlabel("Accuracy")
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_ylabel("Group composition (order-independent)")
    axes[0].set_yticks(np.arange(len(group_order)))
    axes[0].set_yticklabels(group_order)

    # -----------------------------
    # 7. Legend
    # -----------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=marker_map["accuracy"],
            linestyle="None",
            color=main_color,
            alpha=main_alpha,
            markersize=8,
            label="Heterogeneous MAD by model sizes (different points for different debate-order)",
        ),
        Line2D(
            [0],
            [0],
            marker=marker_map["accuracy_best_individual_baseline"],
            linestyle="None",
            color=best_individual_color,
            markersize=8,
            label="Best individual model baseline",
        ),
        Line2D(
            [0],
            [0],
            marker=marker_map["accuracy_no_discussion_voting_same_calls_baseline"],
            linestyle="None",
            color=voting_color,
            markersize=7,
            label="Voting, no discussion baseline",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )

    # -----------------------------
    # 8. Final formatting
    # -----------------------------

    fig.suptitle(
        "Accuracy of Heterogeneous MAD with different model sizes against baselines",
        y=1.10,
        fontsize=14,
    )

    plt.tight_layout()

    return fig
