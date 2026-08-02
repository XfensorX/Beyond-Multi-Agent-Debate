import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from social_groups.analysis.polars_transformations.pareto_front import add_pareto_flag


def pareto_frontier_all_methods(
    different_families_mad_results_combined,
    homogeneous_mad,
    baseline_performance,
    no_discussion_voting_same_size_mixed_families,
    no_discussion_voting_mixed_size_same_family,
    size_heterogeneous_mad,
    *,
    title="Pareto frontier of cost vs. performance",
):
    plot_specs = [
        {
            "df": add_pareto_flag(
                different_families_mad_results_combined,
                [],
                "call_cost_heuristic",
                "accuracy",
                flag_name="mixed_families_pareto",
            ).with_columns(label=pl.col("sizes").list.join("-")),
            "x": "call_cost_heuristic",
            "y": "accuracy",
            "pareto": "mixed_families_pareto",
            "label": "Mixed families MAD ",
            "point_label": "label",  # change if needed
            "group": "MAD",
        },
        {
            "df": add_pareto_flag(
                homogeneous_mad,
                [],
                "call_cost_heuristic",
                "accuracy",
                flag_name="homogeneous_pareto",
            ).with_columns(
                label=pl.col("model_family")
                + pl.lit("-")
                + pl.col("group_constellation")
            ),
            "x": "call_cost_heuristic",
            "y": "accuracy",
            "pareto": "homogeneous_pareto",
            "label": "Homogeneous MAD (original)",
            "point_label": "label_",  # change if needed
            "group": "MAD",
        },
        {
            "df": add_pareto_flag(
                baseline_performance,
                [],
                "call_cost_heuristic",
                "accuracy",
                flag_name="baseline_pareto",
            ).with_columns(
                label=pl.col("model_family") + pl.lit("-") + pl.col("model_letter")
            ),
            "x": "call_cost_heuristic",
            "y": "accuracy",
            "pareto": "baseline_pareto",
            "label": "Baseline (Single Call)",
            "point_label": "group_",  # change if needed
            "group": "Baseline",
        },
        {
            "df": add_pareto_flag(
                no_discussion_voting_same_size_mixed_families,
                [],
                "call_cost_heuristic",
                "accuracy_mean",
                flag_name="no_discussion_mixed_families_same_size_pareto",
            ),
            "x": "call_cost_heuristic",
            "y": "accuracy_mean",
            "pareto": "no_discussion_mixed_families_same_size_pareto",
            "label": "No-discussion voting (Mixed Families Same Capability Levels)",
            "point_label": "model_family_combination",  # change if needed
            "group": "Baseline",
        },
        {
            "df": add_pareto_flag(
                no_discussion_voting_mixed_size_same_family,
                [],
                "call_cost_heuristic",
                "accuracy_mean",
                flag_name="no_discussion_voting_mixed_size_same_family_pareto",
            ).with_columns(label=pl.col("family") + pl.lit("-") + pl.col("group")),
            "x": "call_cost_heuristic",
            "y": "accuracy_mean",
            "pareto": "no_discussion_voting_mixed_size_same_family_pareto",
            "label": "No-discussion voting (Mixed Sizes Same Family)",
            "point_label": "label_",  # change if needed
            "group": "Baseline",
        },
        {
            "df": add_pareto_flag(
                size_heterogeneous_mad,
                [],
                "call_cost_heuristic",
                "accuracy",
                flag_name="size_heterogeneous_pareto",
            ).with_columns(
                label=pl.col("model_family")
                + pl.lit("-")
                + pl.col("group_constellation")
            ),
            "x": "call_cost_heuristic",
            "y": "accuracy",
            "pareto": "size_heterogeneous_pareto",
            "label": "Size - Heterogeneous MAD",
            "point_label": "label_",  # change if needed
            "group": "MAD",
        },
    ]

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.25,
        rc={
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.frameon": True,
        },
    )

    palette = sns.color_palette("muted", n_colors=len(plot_specs))

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    texts = []

    for spec, color in zip(plot_specs, palette):
        df = spec["df"].sort(spec["x"]).to_pandas()

        x = spec["x"]
        y = spec["y"]
        pareto_col = spec["pareto"]
        label_col = spec["point_label"]

        non_pareto = df[~df[pareto_col]]
        pareto = df[df[pareto_col]].sort_values(x)

        ax.plot(
            pareto[x],
            pareto[y],
            color=color,
            linewidth=1.3,
            alpha=0.95,
        )

        ax.scatter(
            pareto[x],
            pareto[y],
            s=42,
            color=color,
            alpha=0.98,
            edgecolor="white",
            linewidth=1.1,
            label=spec["label"],
            zorder=4,
        )

        if label_col in pareto.columns:
            for _, row in pareto.iterrows():
                texts.append(
                    ax.text(
                        row[x],
                        row[y],
                        str(row[label_col]),
                        fontsize=9.5,
                        weight="semibold",
                        ha="right",
                        va="bottom",
                        zorder=5000000,
                    )
                )

    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.2, 1.35),
        expand_text=(1.05, 1.25),
        arrowprops=dict(arrowstyle="-", lw=0.45, alpha=0.45),
    )
    ax.set_title(title, pad=18, fontsize=17)
    ax.set_xlabel("Estimated call cost heuristic")
    ax.set_ylabel("Accuracy")

    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.18)
    ax.minorticks_on()

    handles, labels = ax.get_legend_handles_labels()

    mad_handles = [
        h
        for h, label in zip(handles, labels)
        if plot_specs[labels.index(label)]["group"] == "MAD"
    ]
    mad_labels = [
        label for label in labels if plot_specs[labels.index(label)]["group"] == "MAD"
    ]

    baseline_handles = [
        h
        for h, label in zip(handles, labels)
        if plot_specs[labels.index(label)]["group"] == "Baseline"
    ]
    baseline_labels = [
        label
        for label in labels
        if plot_specs[labels.index(label)]["group"] == "Baseline"
    ]

    baseline_legend = ax.legend(
        baseline_handles,
        baseline_labels,
        title="Baselines",
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.9,
    )

    mad_legend = ax.legend(
        mad_handles,
        mad_labels,
        title="MAD approaches",
        loc="center right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.9,
    )

    baseline_legend.get_title().set_fontweight("bold")
    mad_legend.get_title().set_fontweight("bold")

    ax.add_artist(baseline_legend)

    sns.despine()
    fig.suptitle(
        "Pareto Frontier of Cost vs. Performance",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        "qH - Qwen3 H, QH - Qwen3.5 H, mH - Ministral H",
        fontsize=11,
        color="dimgray",
        pad=12,
    )

    fig.tight_layout()

    return fig
