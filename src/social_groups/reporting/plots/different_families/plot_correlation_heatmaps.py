import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_heatmaps(df: pd.DataFrame):

    metric_order = [
        "accuracy_before",
        "accuracy",
        "accuracy_increase",
        "wd_bias_good proposals",
        "wd_bias_bad proposals",
        "wd_bias_weight on one",
        "wd_bias_False",
    ]

    label_map = {
        "accuracy_before": "Accuracy before",
        "accuracy": "Accuracy after",
        "accuracy_increase": "Accuracy increase",
        "wd_bias_good proposals": "WD good proposals",
        "wd_bias_bad proposals": "WD bad proposals",
        "wd_bias_weight on one": "WD weight on one",
        "wd_bias_False": "WD false",
    }

    display_order = [label_map[x] for x in metric_order]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )
    compositions = [
        "Homogeneous family",
        "Diverse families",
    ]

    for ax, composition in zip(axes, compositions):
        sub = df.loc[
            df["composition_type"] == composition,
            metric_order,
        ].apply(pd.to_numeric, errors="coerce")

        corr = sub.corr(method="pearson")
        corr = corr.loc[metric_order, metric_order]

        corr.index = display_order
        corr.columns = display_order

        sns.heatmap(
            corr,
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="vlag",
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 12},
            cbar=(composition == compositions[-1]),  # only right plot gets colorbar
            cbar_kws={"label": "Pearson correlation"},
        )
        if composition == "Homogeneous family":
            ax.set_ylabel("")
            ax.set_yticklabels([])

        ax.set_title(
            composition,
            fontsize=15,
            fontweight="bold",
            pad=10,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=40,
            ha="right",
        )

        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
        )

        for label in ax.get_yticklabels():
            label.set_horizontalalignment("center")
        ax.tick_params(axis="y", pad=100)

    fig.suptitle(
        "Correlation Structure of Performance and Wasserstein Distance Metrics",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )
    return fig
