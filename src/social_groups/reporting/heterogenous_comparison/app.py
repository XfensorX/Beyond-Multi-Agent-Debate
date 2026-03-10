import json
from functools import reduce
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from social_groups.analysis.polars_transformations import make_group_constellation
from social_groups.analysis.polars_transformations.apply_parsing_and_group_decision import (
    apply_parsing_and_group_decision,
)
from social_groups.directories import REPORTING_DIR
from social_groups.reporting.analysis_columns import AnalysisColumn
from social_groups.reporting.group_decision_scheme import (
    MEMBERS_CORRECT_BEGINNING,
    MEMBERS_CORRECT_END,
    calculate_decision_scheme,
    calculate_extended_decision_scheme,
    extend_by_group_info_before_and_after,
)
from social_groups.reporting.group_reply import (
    GroupReplyAggregator,
    MajorityVote,
    SingularityVote,
)
from social_groups.reporting.parsing import AnswerComparer, AnswerOptions, AnswerParser
from social_groups.reporting.plots.decision_scheme_extended import (
    make_decision_scheme_extended_plot,
)
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo

st.set_page_config(layout="wide")

PHOENIX_BASE_URL = "http://localhost:6006"

DATA_DIR = REPORTING_DIR / "heterogeneous_group"

COLUMN_CONFIG = {
    "phoenix_span_url": st.column_config.LinkColumn(
        display_text=":material/open_in_new:", pinned=True, width="small"
    )
}


st.title("Multi-Agent Debate (MAD) with heterogeneous groups")
st.divider()


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path)


@st.cache_data
def load_mad_dataframe():
    from social_groups.analysis.definitions import defs

    return defs().load_asset_value("hetero_mad")


tabs = st.tabs(["Broad Overview", "Decision Schemes"])


decision_schemes = load_csv(DATA_DIR / "group_decision_schemes.csv")
human_group_overview = load_csv(DATA_DIR / "human_based_table_page_46.csv")
llm_group_overview = load_csv(DATA_DIR / "llm_based_table_page_46.csv")


def make_question_comparison_table_nice(df):
    df_display = df
    for col in df_display.columns:
        if col.startswith("is_correct"):
            df_display = df_display.with_columns(
                pl.when(pl.col(col))
                .then(pl.lit("**✔ Yes**"))
                .otherwise(pl.lit("**✖ No**"))
                .alias(col)
            )
        if col.startswith("phoenix_span_url"):
            df_display = df_display.with_columns(
                pl.col(col)
                .map_elements(
                    lambda url: f"[🔗 link]({url})" if len(url) > 60 else url,
                    return_dtype=pl.String,
                )
                .alias(col)
            )
        if col.startswith("question_id"):
            df_display = df_display.with_columns(pl.col(col).cast(pl.String).alias(col))

    patterns = {
        "question_id": "**Q ID**",
        "is_correct": "**Correct**",
        "phoenix_span_url": "**Phoenix Span**",
        "category": "**Category**",
        "original_question_id": "**Orig. Q ID**",
    }
    pretty_names = {
        key: reduce(lambda s, kv: s.replace(*kv), patterns.items(), key)
        for key in df.columns
    }

    return df_display.rename(pretty_names).to_pandas()


with tabs[0]:
    st.dataframe(decision_schemes)

    def min_max_normalize(df):
        return (df - df.min()) / (df.max() - df.min())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Observed Behaviour of Human Groups")
        human_group_overview["normalized"] = min_max_normalize(
            human_group_overview["score (-115 to 115)"]
        )

        st.dataframe(human_group_overview.sort_values("normalized", ascending=False))

    with col2:
        st.subheader("Observed Relative Performance of LLM based Groups")
        llm_group_overview["normalized"] = min_max_normalize(
            llm_group_overview["accuracy"]
        )
        st.dataframe(llm_group_overview.sort_values("normalized", ascending=False))


with tabs[1]:
    comparison_mode = st.checkbox("Comparison Mode")
    mad_frame = load_mad_dataframe()

    with st.expander("All Data for Multi-Agent Debate"):
        st.dataframe(
            mad_frame,
            column_config=COLUMN_CONFIG,
        )

    selected_constellations = []

    for i, col in enumerate(st.columns(2 if comparison_mode else 1)):
        with col:
            st.subheader("Selection")
            col1, col2, col3 = st.columns(3)

            with col2:
                aggregator = st.selectbox(
                    "Group Aggregator",
                    options=[MajorityVote(), SingularityVote()],
                    format_func=lambda x: {
                        MajorityVote: "Majority Vote",
                        SingularityVote: "Singularity Vote",
                    }[type(x)],
                    key=f"aggregator_{i}",
                )
            with col3:
                triple_underscore_handling = st.selectbox(
                    "Handling of Parsing Errors: ",
                    options=["null", "wrong", "random"],
                    key=f"parsing_handling_{i}",
                )

            comparer = AnswerComparer(
                AnswerOptions.letters_A_to_J,
                triple_underscore_handling=triple_underscore_handling,
            )
            parser = AnswerParser(AnswerOptions.letters_A_to_J)
            group_reply = GroupReplyAggregator(aggregator)

            mad_analysis = apply_parsing_and_group_decision(
                mad_frame, parser, comparer, group_reply
            ).with_columns(make_group_constellation())

            with st.expander("Analyzed Data"):
                st.dataframe(
                    mad_analysis,
                    column_config=COLUMN_CONFIG,
                )

            with col1:
                group = st.selectbox(
                    "Group",
                    options=mad_analysis["group_constellation"]
                    .unique()
                    .sort()
                    .to_list(),
                    key=f"group_{i}",
                )
                selected_constellations.append(group)

            selected_med_analysis = mad_analysis.filter(
                pl.col("group_constellation") == group
            )

            normal_decision_scheme = calculate_decision_scheme(
                selected_med_analysis,
                AnalysisColumn.parsed_individual_answers_before.value,
                AnalysisColumn.parsed_individual_answers_after.value,
                "answer_string",
                group_reply,
                comparer,
            )
            extended_decision_scheme = calculate_extended_decision_scheme(
                selected_med_analysis,
                AnalysisColumn.parsed_individual_answers_before.value,
                AnalysisColumn.parsed_individual_answers_after.value,
                "answer_string",
                group_reply,
                comparer,
            )

            extended_decision_scheme_plot = make_decision_scheme_extended_plot(
                extended_decision_scheme, title=f"Group {group}"
            )

            with st.expander("Normal Decision Scheme"):
                st.table(
                    normal_decision_scheme.select(
                        [
                            "Correct Members Beginning",
                            pl.col("correct").alias("Fraction Correct Group Response"),
                            pl.col("incorrect").alias(
                                "Fraction Incorrect Group Response"
                            ),
                        ]
                    )
                )

                overall_correct = extended_decision_scheme.select(
                    pl.col("correct").dot(pl.col("occurrences"))
                    / pl.col("occurrences").sum()
                ).item()
                st.metric(
                    "Overall Correct",
                    f"{overall_correct:.2f}",
                )
            with st.expander("Extended Decision Scheme"):
                st.dataframe(
                    extended_decision_scheme.drop("incorrect"),
                    key=f"extended_ds_{i}",
                    column_config=COLUMN_CONFIG,
                )

                col1, col2 = st.columns(2)
                with col1:
                    correct_beginning = st.radio(
                        "Members Correct Beginning",
                        extended_decision_scheme["Correct Members Beginning"]
                        .unique()
                        .sort()
                        .to_list(),
                        key=f"correct_beginning_radio_{i}",
                    )
                with col2:
                    correct_end = st.radio(
                        "Members Correct End",
                        extended_decision_scheme["Correct Members End"]
                        .unique()
                        .sort()
                        .to_list(),
                        key=f"correct_end_radio_{i}",
                    )
                st.dataframe(
                    extend_by_group_info_before_and_after(
                        selected_med_analysis,
                        AnalysisColumn.parsed_individual_answers_before.value,
                        AnalysisColumn.parsed_individual_answers_after.value,
                        "answer_string",
                        group_reply,
                        comparer,
                    )
                    .rename(
                        {
                            MEMBERS_CORRECT_BEGINNING: "Correct Members Beginning",
                            MEMBERS_CORRECT_END: "Correct Members End",
                        }
                    )
                    .filter(
                        pl.col("Correct Members Beginning") == correct_beginning,
                        pl.col("Correct Members End") == correct_end,
                    ),
                    column_config=COLUMN_CONFIG,
                )

            with st.expander("Decision Scheme Plot"):
                st.plotly_chart(extended_decision_scheme_plot, key=f"chart_{i}")

            with st.expander("Experiment Configuration"):
                st.json(
                    (
                        selected_med_analysis["experiment_configuration_json"]
                        .unique()
                        .map_elements(
                            lambda x: ExperimentConfig.model_validate(json.loads(x))
                        )
                        .item()
                    ).model_dump_json()
                )

            with st.expander("Meta Info"):
                st.json(
                    (
                        selected_med_analysis["meta_info_json"]
                        .unique()
                        .map_elements(
                            lambda x: ExperimentMetaInfo.model_validate(json.loads(x))
                        )
                        .item()
                    ).model_dump_json()
                )

            with st.expander("All Messages"):
                st.dataframe(
                    selected_med_analysis.drop(
                        [
                            "id",
                            "run_id",
                            "run_identifier",
                            "experiment_id",
                            "experiment_configuration_json",
                            "meta_info_json",
                            "name",
                            "answers_at_beginning",
                            "answers_at_end",
                            "question",
                            "group_constellation",
                            "phoenix_span_id",
                        ]
                    ),
                    column_config=COLUMN_CONFIG,
                    key=f"messages_df_{i}",
                )

    if comparison_mode:
        with st.expander("Compare Answers directly"):
            st.text("Treats not parsable answers as wrong.")
            g1, g2 = selected_constellations

            if g1 == g2:
                st.text("Irrelevant when comparing a group constellation with itself.")
            else:
                group_aggregator = st.selectbox(
                    "Group Aggregator",
                    options=[MajorityVote(), SingularityVote()],
                    format_func=lambda x: {
                        MajorityVote: "Majority Vote",
                        SingularityVote: "Singularity Vote",
                    }[type(x)],
                    key=f"group_aggregator_{i}",
                )

                comparison_analysis = (
                    apply_parsing_and_group_decision(
                        mad_frame,
                        AnswerParser(AnswerOptions.letters_A_to_J),
                        AnswerComparer(
                            AnswerOptions.letters_A_to_J,
                            triple_underscore_handling="wrong",
                        ),
                        GroupReplyAggregator(group_aggregator),
                    )
                    .with_columns(make_group_constellation())
                    .filter(
                        (pl.col("group_constellation") == g1)
                        | (pl.col("group_constellation") == g2)
                    )
                    .unique(
                        subset=[
                            "question_id",
                            "is_correct",
                        ],
                        keep="none",
                    )
                    .select(
                        [
                            "group_constellation",
                            "question_id",
                            "is_correct",
                            "phoenix_span_url",
                            "category",
                            "original_question_id",
                        ]
                    )
                    .sort(["question_id", "group_constellation"])
                )

                combined = (
                    comparison_analysis.filter(pl.col("group_constellation") == g1)
                    .drop("group_constellation")
                    .join(
                        comparison_analysis.filter(
                            pl.col("group_constellation") == g2
                        ).drop("group_constellation"),
                        on=["question_id", "category", "original_question_id"],
                        maintain_order="left",
                        how="inner",
                        validate="1:1",
                        suffix=f"\n{g2}",
                        coalesce=True,
                    )
                    .rename(
                        {
                            "is_correct": f"is_correct\n{g1}",
                            "phoenix_span_url": f"phoenix_span_url \n{g1}",
                        }
                    )
                )

                st.table(
                    make_question_comparison_table_nice(combined),
                    border="horizontal",
                )
