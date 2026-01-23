from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import zstandard as zstd
from config import TRACK_FILE_NAME
from decision_schemes.base import ExampleInput, ExampleOutput
from pydantic import BaseModel
from utils.phoenix import PhoenixExampleHandle

logger = logging.getLogger(__name__)


def compress_zstd(
    src_path: str | Path,
    dst_path: str | Path,
    level: int = 10,
    remove_src: bool = False,
):
    cctx = zstd.ZstdCompressor(level=level)

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        cctx.copy_stream(src, dst)

    logger.info(
        f"Compressed {src_path.name} ({os.path.getsize(src_path) // 1024:,} KB) to {dst_path.name} ({os.path.getsize(dst_path) // 1024:,} KB). "
    )

    if remove_src:
        os.remove(src_path)


class TrackEntry(BaseModel):
    input: ExampleInput
    output: ExampleOutput
    phoenix_span_info: PhoenixExampleHandle


class ExperimentTracker:
    def __init__(self, output_directory: Path):
        self.out_path = output_directory / TRACK_FILE_NAME
        self.out_path_compressed = output_directory / (TRACK_FILE_NAME + ".zst")

    def __enter__(self):
        self.f = open(self.out_path, "w", encoding="utf-8")
        return self

    def write_line(self, entry: TrackEntry):
        self.f.write(
            json.dumps(entry.model_dump(), ensure_ascii=False, indent=None) + "\n"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()

        if exc_type:
            logger.info("Tracked partial results.")

        logger.info("Compressing results...")
        compress_zstd(self.out_path, self.out_path_compressed, remove_src=True)

        return False
