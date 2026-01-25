import os
from pathlib import Path

import polars as pl
import streamlit as st

from social_groups.trialrunner import PhoenixExampleHandle

PHOENIX_BASE_URL = "http://localhost:6006"
RESULTS_DIR = "/Users/philipp/Documents/Studium/Informatik/Masterthesis/Repository/social_studies/sandbox/results"

PARQUET_FILES = {
    "questions": RESULTS_DIR + "/questions.parquet",
    "answers": RESULTS_DIR + "/answers.parquet",
    "experiments": RESULTS_DIR + "/experiments.parquet",
    "runs": RESULTS_DIR + "/runs.parquet",
}

COLUMN_CONFIGS = {
    "questions": {},
    "answers": {"phoenix_span_id": st.column_config.LinkColumn()},
    "experiments": {},
    "runs": {
        "execution_config_json": st.column_config.JsonColumn(width="large"),
        "meta_config_json": st.column_config.JsonColumn(width="large"),
    },
}


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def show_tab(name: str, path: str) -> None:
    st.subheader(name)
    st.caption(path)

    if not Path(path).exists():
        st.error(f"File not found: {path}")
        return

    df = load_parquet(path)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.height:,}")
    c2.metric("Columns", f"{df.width:,}")
    c3.metric("Size", f"{os.path.getsize(path) / (1024 * 1024):.2f} MB")

    with st.expander("Schema", expanded=False):
        st.write(df.schema)

    # Only show slider if there is a real range
    if df.height <= 100:
        max_rows = df.height
        st.caption(f"Showing all {df.height:,} rows (small table)")
    else:
        max_rows = st.slider(
            "Rows to display (preview)",
            min_value=100,
            max_value=min(200_000, df.height),
            value=min(10_000, df.height),
            step=100,
        )

    if name == "answers":
        df = df.with_columns(
            pl.struct(["phoenix_span_id", "run_identifier"])
            .map_elements(
                lambda s: PhoenixExampleHandle(
                    span_id_hex=s["phoenix_span_id"]
                ).span_url(
                    PHOENIX_BASE_URL,
                    s["run_identifier"],
                ),
                return_dtype=pl.Utf8,
            )
            .alias("phoenix_span_id")
        )

    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        hide_index=True,
        column_config=COLUMN_CONFIGS[name],
    )


def main() -> None:
    st.set_page_config(page_title="Parquet Viewer", layout="wide")
    st.title("Parquet Viewer")

    tab_names = list(PARQUET_FILES.keys())
    tabs = st.tabs(tab_names)

    for tab, name in zip(tabs, tab_names):
        with tab:
            show_tab(name, PARQUET_FILES[name])


if __name__ == "__main__":
    main()
