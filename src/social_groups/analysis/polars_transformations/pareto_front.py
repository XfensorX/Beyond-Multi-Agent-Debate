import polars as pl


def add_pareto_flag(
    df: pl.DataFrame,
    group_cols: list[str],
    cost_col: str,
    performance_col: str,
    *,
    flag_name: str = "is_pareto",
) -> pl.DataFrame:

    def mark_group(group: pl.DataFrame) -> pl.DataFrame:
        sorted_group = group.sort(
            [cost_col, performance_col],
            descending=[False, True],
        )
        best_acc = float("-inf")
        pareto_rows = []

        for row in sorted_group.iter_rows(named=True):
            if row[performance_col] > best_acc:
                pareto_rows.append(True)
                best_acc = row[performance_col]
            else:
                pareto_rows.append(False)

        sorted_group = sorted_group.with_columns(
            pl.Series(name=flag_name, values=pareto_rows)
        )
        return sorted_group

    if group_cols:
        return df.group_by(group_cols).map_groups(mark_group)
    else:
        return mark_group(df)
