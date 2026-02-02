import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from social_groups.reporting.plots.config import PAPER_COLORSCALE, TEXT_FONT


def calculate_heatmap_ys(heights: list[float]):
    heights = np.array(heights)[::-1]
    heights_norm = heights / heights.sum()

    # Compute cumulative edges
    y_edges = 1 - np.concatenate([[0], np.cumsum(heights_norm)])

    # Compute row centers
    ys = (y_edges[:-1] + y_edges[1:]) / 2
    return ys


def _make_heatmap(
    table: np.typing.NDArray[np.float64], heights_top_to_bottom: list[int]
):
    row_idx = np.repeat(
        np.arange(table.shape[0])[::-1].reshape(-1, 1),
        table.shape[1],
        axis=1,
    )

    return go.Heatmap(
        z=table,
        x=np.arange(table.shape[1]),  # equal-width columns
        y=calculate_heatmap_ys(heights_top_to_bottom),
        colorscale=PAPER_COLORSCALE,
        zmin=0,
        zmax=1,
        showscale=True,
        xgap=0,
        ygap=1,
        text=np.round(table.astype(float), 2),
        texttemplate="%{text:.2f}",
        customdata=np.dstack((row_idx,)),
        textfont=TEXT_FONT,
        hovertemplate=(
            "Fraction of cases with group correct at end: %{z:.2f} <br>"
            "Members correct at Start:  %{customdata[0]}<extra></extra>"
        ),
    )


def make_decision_scheme_extended_plot(
    extended_decision_scheme: pl.DataFrame, title: str = "Extended Decision Scheme"
) -> go.Figure:
    all_cases = list(
        sorted(
            set(extended_decision_scheme["Correct Members Beginning"].unique()).union(
                set(extended_decision_scheme["Correct Members End"].unique())
            ),
            reverse=False,
        )
    )

    left_groups = [f" {x}☑ (Start)" for x in all_cases]
    right_groups = [f" {x}☑ (End)" for x in all_cases]

    correct_at_end_given_correct_start_members = (
        pl.DataFrame({"Correct Members Beginning": all_cases})
        .join(
            extended_decision_scheme.group_by("Correct Members Beginning")
            .agg(
                correct=(
                    pl.col("correct").dot(pl.col("occurrences")) / pl.sum("occurrences")
                ).mean()
            )
            .select("correct", "Correct Members Beginning"),
            on="Correct Members Beginning",
            how="left",
        )
        .with_columns(pl.col("correct").fill_null(0))
        .sort("Correct Members Beginning", descending=True)
        .select("correct")
        .to_numpy()
        .reshape(len(right_groups), 1)
    )

    links = (
        extended_decision_scheme.select(
            "Correct Members Beginning", "Correct Members End", "occurrences"
        )
        .map_rows(lambda r: np.array(r))["map"]
        .to_list()
    )

    nL = len(left_groups)

    source = [s for s, t, v in links]
    target = [nL + t for s, t, v in links]
    value = [v for s, t, v in links]

    labels = left_groups + right_groups

    node_colors = ["#A3B1C6", "#B9A3C9", "#323647", "#D4A35C", "#6C7A89"][
        : len(left_groups)
    ] * 2

    # Link colors: use left color with alpha
    def hex_to_rgba(hex_color, a=0.22):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    link_colors = [hex_to_rgba(node_colors[s], 0.22) for s in source]

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 2, 0.4],
        specs=[[{"type": "heatmap"}, {"type": "sankey"}, {"type": "heatmap"}]],
        horizontal_spacing=0.01,
        subplot_titles=[
            "Correct Group Dec.<br> (p. No. corr. @ start)",
            title + "<br> ",
            "",
        ],
    )

    fig.add_trace(
        _make_heatmap(
            correct_at_end_given_correct_start_members,
            [
                sum([v for s, t, v in links if source_ == s])
                for source_ in range(len(left_groups))
            ],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=20,
                thickness=16,
                color=node_colors,
                line=dict(color="rgba(0,0,0,0.3)", width=0.6),
                # Change the order ot the nodes ...
                x=[0.01 for _ in left_groups] + [0.99 for _ in right_groups],
                y=np.cumsum([0.17 for _ in left_groups]).tolist()[::-1] * 2,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
            ),
            textfont=TEXT_FONT,
        ),
        row=1,
        col=2,
    )

    # fig.add_trace(
    #     _make_heatmap(
    #         right_table,
    #         [
    #             sum([v for s, t, v in links if target_ == t])
    #             for target_ in range(len(right_groups))
    #         ],
    #     ),
    #     row=1,
    #     col=3,
    # )

    fig.update_layout(
        margin=dict(l=60, r=20, t=50, b=10),
    )

    # Clean axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    fig.update_legends(font=TEXT_FONT)

    return fig
