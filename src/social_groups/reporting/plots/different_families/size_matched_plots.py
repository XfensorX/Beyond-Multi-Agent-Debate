import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_size_matched_performance(df: pd.DataFrame, PERF_COLS):
    long_df = df.melt(
        id_vars=["composition_type", "size_level"],
        value_vars=PERF_COLS,
        var_name="Metric",
        value_name="Value",
    )

    metric_labels = {
        "accuracy_before": "Accuracy before",
        "accuracy": "Accuracy after",
        "accuracy_increase": "Accuracy increase",
    }

    long_df["Metric"] = long_df["Metric"].map(metric_labels)

    g = sns.catplot(
        data=long_df,
        x="size_level",
        y="Value",
        hue="composition_type",
        col="Metric",
        kind="bar",
        errorbar=("ci", 95),
        height=3.2,
        aspect=0.95,
        sharey=False,
        capsize=0.12,
        err_kws={"linewidth": 1.2},
    )

    g.set_axis_labels("Model size group", "Score")
    g.set_titles("{col_name}")
    g.legend.set_title("Composition")

    for ax in g.axes.flat:
        ax.set_xlabel("Model size group")
        ax.grid(axis="y", alpha=0.3)

    g.figure.suptitle(
        "Size-matched performance comparison",
        y=1.08,
        fontsize=15,
        fontweight="bold",
    )
    return g.figure


def plot_size_matched_wd(df: pd.DataFrame, WD_COLS):
    long_df = df.melt(
        id_vars=["composition_type", "size_level"],
        value_vars=WD_COLS,
        var_name="Metric",
        value_name="Wasserstein distance",
    )

    metric_labels = {
        "wd_bias_good proposals": "Bias: Good proposals",
        "wd_bias_bad proposals": "Bias: Bad proposals",
        "wd_bias_weight on one": "Bias: Weight on one",
        "wd_bias_False": "No Bias",
    }

    long_df["Metric"] = long_df["Metric"].map(metric_labels)

    g = sns.catplot(
        data=long_df,
        x="size_level",
        y="Wasserstein distance",
        hue="composition_type",
        col="Metric",
        kind="bar",
        errorbar=("ci", 95),
        height=3.2,
        aspect=0.95,
        sharey=False,
        capsize=0.12,
        err_kws={"linewidth": 1.2},
    )

    g.set_axis_labels("Model size group", "Wasserstein distance")
    g.set_titles("{col_name}")
    g.legend.set_title("Composition")

    for ax in g.axes.flat:
        ax.set_xlabel("Model size group")
        ax.grid(axis="y", alpha=0.3)

    g.figure.suptitle(
        "Size-matched Wasserstein distance comparison",
        y=1.08,
        fontsize=15,
        fontweight="bold",
    )

    return g.figure
