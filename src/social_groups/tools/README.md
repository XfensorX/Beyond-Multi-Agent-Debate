# `tools`

`tools` contains small Streamlit utilities for manual inspection of generated experiment artifacts.

## Entry Point

```shell
uv run task tools
```

The task runs:

```shell
uv run streamlit run src/social_groups/tools/app.py
```

## Available Pages

| Page | Purpose |
| --- | --- |
| `pages/1_Track_File_Viewer.py` | Upload and inspect a compressed `experiment_result.jsonl.zst` track file. |

## Track File Viewer

The viewer:

1. Accepts a `.zst` or `.jsonl.zst` upload.
2. Decompresses the zstandard stream.
3. Parses each line as JSON.
4. Validates each row as `TrackEntry`.
5. Displays searchable rows with question text, output status, final answer, token counts, and history length.

This is a debugging and inspection tool. It does not write analysis outputs.

This documentation was generated using an LLM
