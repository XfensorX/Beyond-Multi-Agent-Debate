import polars as pl

"""
Generated Using Grok.
"""


def assert_one_to_one_mapping(df: pl.DataFrame, col1: str, col2: str) -> None:
    """
    Raises AssertionError if the two columns do not form a strict 1-1 mapping.
    """
    # 1. Check col1 → col2 consistency (one col1 value → only one col2)
    conflicts1 = (
        df.group_by(col1)
        .agg(pl.col(col2).n_unique().alias("n_unique_col2"))
        .filter(pl.col("n_unique_col2") > 1)
    )

    # 2. Check col2 → col1 consistency (one col2 value → only one col1)
    conflicts2 = (
        df.group_by(col2)
        .agg(pl.col(col1).n_unique().alias("n_unique_col1"))
        .filter(pl.col("n_unique_col1") > 1)
    )

    # 3. Optional: also verify that the number of unique pairs equals the number of unique values
    #    (this catches the case where A→X and B→X even if the above didn't trigger due to some edge cases)
    n_unique_pairs = df.select(
        pl.concat_str([col1, pl.lit("→"), col2]).alias("pair")
    ).n_unique()
    n_unique_col1 = df[col1].n_unique()
    n_unique_col2 = df[col2].n_unique()

    error_msg = []

    if conflicts1.height > 0:
        error_msg.append(f"col1 → col2 mapping is not unique:\n{conflicts1}")

    if conflicts2.height > 0:
        error_msg.append(f"col2 → col1 mapping is not unique:\n{conflicts2}")

    if n_unique_pairs != n_unique_col1 or n_unique_pairs != n_unique_col2:
        error_msg.append(
            f"Mapping is not bijective. "
            f"Unique pairs: {n_unique_pairs}, "
            f"Unique {col1}: {n_unique_col1}, "
            f"Unique {col2}: {n_unique_col2}"
        )

    if error_msg:
        raise AssertionError("\n".join(error_msg))

    print(f"✅ Assertion passed: {col1} ↔ {col2} is a strict 1-1 mapping.")
