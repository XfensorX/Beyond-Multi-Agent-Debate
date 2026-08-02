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


def join_equal_suffix_cols(
    df: pl.DataFrame,
    suffix: str = "_right",
    null_equals_value: bool = False,
) -> pl.DataFrame:

    right_cols = [c for c in df.columns if c.endswith(suffix)]

    to_remove: list[str] = []

    updates: list[pl.Expr] = []

    for right_col in right_cols:
        base_col = right_col.removesuffix(suffix)

        if base_col not in df.columns:
            continue

        if null_equals_value:
            mismatch_expr = (
                pl.col(base_col).is_not_null()
                & pl.col(right_col).is_not_null()
                & (pl.col(base_col) != pl.col(right_col))
            )

        else:
            mismatch_expr = ~pl.col(base_col).eq_missing(pl.col(right_col))

        has_mismatch = df.select(mismatch_expr.any()).item()

        if not has_mismatch:
            if null_equals_value:
                updates.append(
                    pl.coalesce([pl.col(base_col), pl.col(right_col)]).alias(base_col)
                )

            to_remove.append(right_col)

    if updates:
        df = df.with_columns(updates)

    return df.drop(to_remove)
