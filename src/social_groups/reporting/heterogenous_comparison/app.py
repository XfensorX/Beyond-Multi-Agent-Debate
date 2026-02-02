import json
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from social_groups.directories import REPORTING_DIR
from social_groups.reporting.analysis_columns import AnalysisColumn
from social_groups.reporting.group_decision_scheme import (
    calculate_decision_scheme,
    calculate_extended_decision_scheme,
)
from social_groups.reporting.group_reply import (
    GroupReplyAggregator,
    MajorityVote,
    SingularityVote,
)
from social_groups.reporting.heterogenous_comparison.data_retrieval import (
    apply_parsing_and_group_decision,
    get_mad_frame,
)
from social_groups.reporting.parsing import AnswerComparer, AnswerOptions, AnswerParser
from social_groups.reporting.plots.decision_scheme_extended import (
    make_decision_scheme_extended_plot,
)
from social_groups.trialrunner.utils.hydra_config import ExperimentConfig
from social_groups.trialrunner.utils.meta_info import ExperimentMetaInfo
from social_groups.trialrunner.utils.phoenix import (
    retrieve_phoenix_url_from_span_id,
)

st.set_page_config(layout="wide")

PHOENIX_BASE_URL = "http://localhost:6006"

DATA_DIR = REPORTING_DIR / "heterogeneous_group"


st.title("Multi-Agent Debate (MAD) with heterogeneous groups")
st.divider()


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path)


tabs = st.tabs(["Broad Overview", "Decision Schemes"])


decision_schemes = load_csv(DATA_DIR / "group_decision_schemes.csv")
human_group_overview = load_csv(DATA_DIR / "human_based_table_page_46.csv")
llm_group_overview = load_csv(DATA_DIR / "llm_based_table_page_46.csv")


@st.cache_data
def get_span_url(span_id: str):
    return retrieve_phoenix_url_from_span_id(
        span_id,
        phoenix_base_url=PHOENIX_BASE_URL,
    )


def set_selected_span_id(span_id: str | None):
    if st.session_state["SELECTED_PHOENIX_SPAN_ID"] == span_id:
        return

    if span_id is None:
        st.session_state["SELECTED_PHOENIX_SPAN_ID"] = None
        st.session_state["SELECTED_PHOENIX_SPAN_URL"] = None
        return

    st.session_state["SELECTED_PHOENIX_SPAN_ID"] = span_id
    st.session_state["SELECTED_PHOENIX_SPAN_URL"] = get_span_url(span_id)


set_selected_span_id(None)

with st.sidebar:
    set_selected_span_id("df570665ee8d0095")

    if st.session_state["SELECTED_PHOENIX_SPAN_URL"]:
        st.link_button(
            "Open Phoenix Span", url=st.session_state["SELECTED_PHOENIX_SPAN_URL"]
        )

    comparison_mode = st.checkbox("Comparison Mode")

with tabs[0]:
    set_selected_span_id(None)
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
    mad_frame = st.cache_data(get_mad_frame)()

    with st.expander("All Data for Multi-Agent Debate"):
        st.dataframe(mad_frame)

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
            )

            with st.expander("Analyzed Data"):
                st.dataframe(mad_analysis)

            with col1:
                group = st.selectbox(
                    "Group",
                    options=mad_analysis["group_constellation"]
                    .unique()
                    .sort()
                    .to_list(),
                    key=f"group_{i}",
                )

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

            st.subheader("Normal Decision Scheme")
            st.table(
                normal_decision_scheme.select(
                    [
                        "Correct Members Beginning",
                        pl.col("correct").alias("Fraction Correct Group Response"),
                        pl.col("incorrect").alias("Fraction Incorrect Group Response"),
                    ]
                ).to_pandas(),
            )
            st.subheader("Extended Decision Scheme")
            st.dataframe(
                extended_decision_scheme.drop("incorrect").to_pandas(),
                key=f"extended_ds_{i}",
            )
            st.subheader("Decision Scheme Plot")
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

            with st.expander("Messages"):
                selection = st.dataframe(
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
                        ]
                    ),
                    selection_mode="single-row",
                    on_select="rerun",
                    key=f"messages_df_{i}",
                )
            if selection.selection.rows:
                set_selected_span_id(
                    selected_med_analysis[selection.selection.rows[0]][
                        "phoenix_span_id"
                    ].item()
                )
