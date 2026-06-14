import polars as pl


def assert_suffix_cols_equal(
    df: pl.DataFrame,
    suffix: str = "_right",
) -> pl.DataFrame:
    right_cols = [c for c in df.columns if c.endswith(suffix)]
    for right_col in right_cols:
        base_col = right_col.removesuffix(suffix)
        if base_col not in df.columns:
            continue
        mismatch = df.filter(pl.col(base_col).ne_missing(pl.col(right_col)))
        if not mismatch.is_empty():
            raise ValueError(f"Column mismatch: {base_col} != {right_col}")
    return df.drop(right_cols)
