# `reporting`

`reporting` contains reusable evaluation and visualization helpers. These functions are used by Dagster assets and notebooks after the raw experiment data has been parsed into tabular form.

## Responsibilities

- Parse model outputs into multiple-choice answer letters.
- Aggregate individual answers into group decisions.
- Compare predictions against target answers.
- Define reporting-oriented column names.
- Build plots for baseline, voting, homogeneous MAD, heterogeneous MAD, diversity, prompting, and family-comparison analyses.

## Important Files

| Path | Purpose |
| --- | --- |
| `parsing.py` | Regex-based answer parsers and answer comparison logic. |
| `group_reply.py` | Group aggregation strategies such as singularity and majority vote. |
| `group_decision_scheme.py` | Helpers around group reply aggregation and parsed answers. |
| `analysis_columns.py` | Enum of analysis column names used across transformations. |
| `decision_scheme_comparison.py` | Decision-scheme comparison helpers. |
| `theoretical_decision_schemes.py` | Theoretical decision scheme utilities. |
| `plots/` | Matplotlib, Seaborn, and Plotly figure functions. |
| `heterogenous_comparison/app.py` | Streamlit app for interactive heterogeneous comparison analysis. |

## Parsing and Aggregation

`AnswerParser` supports answer-option sets:

- `letters_A_to_J`
- `letters_A_to_D`
- `letters_A_to_J_naive`

`AnswerComparer` compares parsed answers to targets and controls handling of parser errors with `null`, `wrong`, or `random`.

`GroupReplyAggregator` wraps a group reply strategy for use inside Polars expressions. Implemented strategies include:

- `SingularityVote`: valid only if all non-invalid votes agree.
- `MajorityVote`: picks the majority among non-invalid votes and marks ties as different votes.

## Inputs and Outputs

Inputs are usually Polars `DataFrame` or `Expr` objects produced by the `analysis` package. Outputs are transformed frames, scalar columns, or Matplotlib/Seaborn/Plotly figure objects that notebooks materialize as report assets.

## Plot Modules

The `plots/` directory contains focused plotting functions rather than command-line entry points. Notebooks call these functions and then register generated figures through the analysis notebook asset helpers.

This documentation was generated using an LLM
