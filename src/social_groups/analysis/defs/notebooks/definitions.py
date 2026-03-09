from pathlib import Path
from typing import Any

import dagster as dg
import dagstermill
import pandas as pd
import polars as pl
from dagstermill import define_dagstermill_asset
from pydantic import BaseModel

from social_groups.directories import DAGSTER_BASE_DIR


class ExtraNotebookAsset(BaseModel):
    name: str
    extension: str

    def get_key(self):
        return dg.AssetKey(["report", self.name])

    def get_spec(self, *, deps):
        return dg.AssetSpec(
            key=self.get_key(),
            deps=deps,
            group_name="reports",
            metadata={"file_extension": self.extension},
        )

    def _get_file_name(self) -> str:
        return f"{self.get_key().path[-1]}.{self.extension}"

    def _get_path(self) -> Path:
        return (
            DAGSTER_BASE_DIR / Path(*self.get_key().path[:-1]) / self._get_file_name()
        )

    def register_materialization(self, obj: Any, description: str):
        """Used from within a notebook to register the materialization of output assets."""
        path = self._get_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, str):
            path.write_text(obj)
        elif self.extension == "csv":
            if isinstance(obj, pl.DataFrame):
                obj.to_pandas().to_csv(path, index=False)
            elif isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=False)
            else:
                raise NotImplementedError()
        elif self.extension == "tex":
            if isinstance(obj, pl.DataFrame):
                obj.to_pandas().to_latex(path, index=False)
            elif isinstance(obj, pd.DataFrame):
                obj.to_latex(path, index=False)
            else:
                raise NotImplementedError()
        else:
            raise ValueError(f"Cannot handle type {type(obj)}")

        dagstermill.yield_event(
            dg.AssetMaterialization(
                asset_key=self.get_key(),
                description=description,
                metadata={
                    "path": dg.MetadataValue.path(path),
                    "file_name": dg.MetadataValue.text(self.name),
                    "size_bytes": path.stat().st_size,
                    "file_extension": self.extension,
                },
            )
        )


def create_notebook_asset(
    file_name: str,
    ins: dict[str, dg.AssetIn],
    extra_assets: list[ExtraNotebookAsset] | None = None,
) -> list[dg.AssetsDefinition | dg.OpDefinition]:
    notebook_def = define_dagstermill_asset(
        key_prefix=["notebooks"],
        name=file_name.split(".")[0],
        notebook_path=dg.file_relative_path(__file__, file_name),
        group_name="notebooks",
        ins=ins,
        metadata={
            "Original Notebook": dg.MetadataValue.notebook(
                dg.file_relative_path(__file__, file_name)
            ),
            "dagster/code_references": dg.CodeReferencesMetadataValue(
                code_references=[
                    dg.LocalFileCodeReference(
                        file_path=dg.file_relative_path(__file__, file_name),
                        line_number=0,
                        label="Original Notebook",
                    ),
                    dg.LocalFileCodeReference(
                        file_path=str(DAGSTER_BASE_DIR / file_name),
                        line_number=0,
                        label="Run Notebook",
                    ),
                ]
            ),
        },
    )

    if extra_assets is None:
        return [notebook_def]

    return [notebook_def] + [
        spec.get_spec(deps=[notebook_def]) for spec in extra_assets
    ]


defs = dg.Definitions(
    assets=[
        *create_notebook_asset(
            "baseline_analysis.ipynb",
            ins={"baseline_frame": dg.AssetIn("baseline")},
        ),
        *create_notebook_asset(
            "heterogeneous_comparison.ipynb",
            ins={
                "mad_frame": dg.AssetIn("hetero_mad"),
                "baseline_frame": dg.AssetIn("baseline"),
            },
        ),
        *create_notebook_asset(
            "thinking_mad_analysis.ipynb",
            ins={"thinking_frame": dg.AssetIn("thinking_mad")},
        ),
        *create_notebook_asset(
            "diversity_params_analysis.ipynb",
            ins={"diversity_frame": dg.AssetIn("diversity_params_mad")},
        ),
        *create_notebook_asset(
            "changed_order_mad_analysis.ipynb",
            ins={"frame": dg.AssetIn("changed_order_mad")},
        ),
        *create_notebook_asset(
            "changed_prompt_mad_analysis.ipynb",
            ins={
                "changed_prompt": dg.AssetIn("changed_prompt_mad"),
                "original_prompt": dg.AssetIn("hetero_mad"),
            },
        ),
        *create_notebook_asset(
            "no_discussion_voting_analysis.ipynb",
            ins={"frame": dg.AssetIn("no_discussion_voting")},
        ),
        *create_notebook_asset(
            "baseline_output_comparison_analysis.ipynb",
            ins={"baseline_output_frame": dg.AssetIn("baseline_output_comparison")},
            extra_assets=[
                ExtraNotebookAsset(name="a_test_table_in_latex", extension="tex")
            ],
        ),
    ],
)
