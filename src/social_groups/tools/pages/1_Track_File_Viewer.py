import io
import json

import pandas as pd
import streamlit as st
import zstandard as zstd
from pydantic import ValidationError

from social_groups.general.tracking import TrackEntry


def decompress_zst_and_yield_lines(data: bytes):
    """Decompress zstd stream and yield lines one by one"""
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(data)) as reader:
        buffer = b""
        while True:
            chunk = reader.read(65536)
            if not chunk:
                if buffer:
                    yield buffer.decode("utf-8", errors="replace")
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace")


@st.cache_data(show_spinner="Parsing compressed JSONL…", persist=False)
def parse_track_file(uploaded_bytes: bytes) -> list[dict]:
    valid_entries = []
    invalid_count = 0

    for i, line in enumerate(decompress_zst_and_yield_lines(uploaded_bytes)):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            entry = TrackEntry.model_validate(data)
            valid_entries.append(
                {
                    "idx": i,
                    "question": entry.input.question,
                    "presolved": entry.input.presolved_questions[:120] + "…"
                    if len(entry.input.presolved_questions) > 120
                    else entry.input.presolved_questions,
                    "has_output": entry.output is not None,
                    "final_answer": (entry.output.final_answer or "—")[:80] + "…"
                    if entry.output and entry.output.final_answer
                    else "—",
                    "input_tokens": entry.output.used_input_tokens
                    if entry.output
                    else 0,
                    "output_tokens": entry.output.used_output_tokens
                    if entry.output
                    else 0,
                    "history_len": len(entry.output.history) if entry.output else 0,
                }
            )
        except (json.JSONDecodeError, ValidationError) as e:
            invalid_count += 1
            if invalid_count <= 3:
                st.warning(f"Line {i + 1} failed validation: {str(e)[:120]}")

    if invalid_count > 3:
        st.warning(f"… and {invalid_count - 3} more invalid entries were skipped.")

    return valid_entries


st.set_page_config(page_title="Track File Viewer", layout="wide")
st.title("Track File Viewer  (.jsonl.zst)")

st.markdown("Drag & drop a **compressed JSONL file** (`.jsonl.zst`)")

uploaded_file = st.file_uploader(
    "Upload track file",
    type=["zst", "jsonl.zst"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    with st.spinner("Decompressing and parsing..."):
        records = parse_track_file(file_bytes)

    if not records:
        st.error("No valid entries found in the file.")
        st.stop()

    df = pd.DataFrame(records)

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        search = st.text_input("Search question / answer", "")
    with col2:
        has_output = st.radio(
            "Output exists?",
            options=["All", "With output", "Without output"],
            horizontal=True,
            index=0,
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        rows = st.slider("Rows per page", 5, 50, 15, step=5)

    filtered_df = df.copy()

    if search:
        mask = filtered_df["question"].str.contains(
            search, case=False, na=False
        ) | filtered_df["final_answer"].str.contains(search, case=False, na=False)
        filtered_df = filtered_df[mask]

    if has_output == "With output":
        filtered_df = filtered_df[filtered_df["has_output"]]
    elif has_output == "Without output":
        filtered_df = filtered_df[~filtered_df["has_output"]]

    # ── Styling & display ───────────────────────────────────
    st.dataframe(
        filtered_df,
        column_config={
            "idx": st.column_config.NumberColumn("Idx", width="small"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "presolved": st.column_config.TextColumn(
                "Presolved (short)", width="medium"
            ),
            "final_answer": st.column_config.TextColumn("Final Answer", width="large"),
            "input_tokens": "In Tokens",
            "output_tokens": "Out Tokens",
            "history_len": "# Turns",
            "has_output": st.column_config.CheckboxColumn("Has Output?"),
        },
        hide_index=True,
        use_container_width=True,
        height=38 + 38 * min(rows, len(filtered_df)),
    )

    st.caption(
        f"Showing {len(filtered_df)} of {len(df)} entries • {len(df) - len(filtered_df)} filtered out"
    )

else:
    st.info("Upload a .jsonl.zst file to begin")
