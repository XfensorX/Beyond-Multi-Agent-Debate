import pprint
import subprocess
from pathlib import Path
from typing import Any, Literal

import altair
import dagster as dg
import dagstermill
import pandas as pd
import polars as pl
import seaborn
from dagstermill import define_dagstermill_asset
from matplotlib.figure import Figure
from pydantic import BaseModel

from social_groups.directories import DAGSTER_BASE_DIR

SUPPORTED_EXTENSION = Literal["csv", "tex", "svg", "pdf", "png"]


class ExtraNotebookAsset(BaseModel):
    name: str
    extension: SUPPORTED_EXTENSION
    notebook_name: str

    def get_key(self):
        return dg.AssetKey(
            ["report", f"{self.notebook_name.replace('.ipynb', '')}", self.name]
        )

    def get_spec(self, *, deps) -> dg.AssetSpec:
        return dg.AssetSpec(
            key=self.get_key(),
            deps=deps,
            group_name=self.notebook_name.replace(".ipynb", ""),
            metadata={"file_extension": self.extension},
            description="See Metadata for description.",
        )

    def _get_file_name(self) -> str:
        return f"{self.get_key().path[-1]}.{self.extension}"

    def _get_path(self) -> Path:
        return (
            DAGSTER_BASE_DIR / Path(*self.get_key().path[:-1]) / self._get_file_name()
        )

    def _get_preview(self, obj: Any) -> dg.MetadataValue | None:
        if isinstance(obj, str):
            return dg.MetadataValue.text(obj)
        elif self.extension == "csv":
            if isinstance(obj, pl.DataFrame):
                return dg.MetadataValue.md(obj.to_pandas().to_markdown(index=False))
            elif isinstance(obj, pd.DataFrame):
                return dg.MetadataValue.md(obj.to_markdown(index=False))
            else:
                raise NotImplementedError
        elif self.extension == "tex":
            if isinstance(obj, pl.DataFrame):
                return dg.MetadataValue.md(obj.to_pandas().to_markdown(index=False))
            elif isinstance(obj, pd.DataFrame):
                return dg.MetadataValue.md(obj.to_markdown(index=False))
            else:
                raise NotImplementedError
        elif self.extension in {"svg", "pdf", "png"}:
            return None  # TODO: does pdf work here maybe?
        else:
            raise ValueError(f"Cannot handle type {type(obj)}")

    def register_materialization(
        self, obj: Any, description: str, print_output_path: bool = False
    ):
        """Used from within a notebook to register the materialization of output assets."""
        path = self._get_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if print_output_path:
            print("Wrote to: ", path)

        if isinstance(obj, str):
            path.write_text(obj)
        elif self.extension == "csv":
            if isinstance(obj, pl.DataFrame):
                obj.to_pandas().to_csv(path, index=False)
            elif isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=False)
            else:
                raise NotImplementedError
        elif self.extension == "tex":
            if isinstance(obj, pl.DataFrame):
                obj = obj.rename({col: col.replace("_", " ") for col in obj.columns})

                obj.to_pandas().to_latex(
                    path, index=False, float_format="{:,.1f}".format
                )
            elif isinstance(obj, pd.DataFrame):
                obj.to_latex(path, index=False)
            elif isinstance(obj, str):
                with open(path, "w") as f:
                    f.write(obj)
            else:
                raise NotImplementedError
        elif self.extension in {"svg", "pdf", "png"}:
            if isinstance(obj, Figure):
                obj.savefig(path)
            elif isinstance(obj, altair.vegalite.v6.api.Chart):
                obj.save(path)
            elif isinstance(obj, seaborn.FacetGrid):
                obj.savefig(path)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Cannot handle type {type(obj)}")

        dagstermill.yield_event(
            dg.AssetMaterialization(
                asset_key=self.get_key(),
                description=description,
                metadata={
                    "description": description,
                    "path": dg.MetadataValue.path(path),
                    "file_name": dg.MetadataValue.text(self.name),
                    "size_bytes": path.stat().st_size,
                    "file_extension": self.extension,
                    "preview": self._get_preview(obj),
                },
            )
        )


def create_notebook_asset(
    file_name: str,
    ins: dict[str, dg.AssetIn],
    extra_assets: list[ExtraNotebookAsset] | None = None,
) -> list[dg.AssetsDefinition | dg.AssetSpec]:
    notebook_path = dg.file_relative_path(
        __file__, str(Path("defs/notebooks") / file_name)
    )

    if not Path(notebook_path).exists():
        raise FileNotFoundError(notebook_path)

    notebook_def = define_dagstermill_asset(
        key_prefix=["notebooks"],
        name=file_name.split(".")[0],
        notebook_path=notebook_path,
        group_name="notebooks",
        ins=ins,
        metadata={
            "Original Notebook": dg.MetadataValue.notebook(notebook_path),
            "dagster/code_references": dg.CodeReferencesMetadataValue(
                code_references=[
                    dg.LocalFileCodeReference(
                        file_path=notebook_path,
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


def code_to_latex_snippet(
    code: str, *, caption: str, label: str, style="python"
) -> str:
    def format_string_with_ruff(source_code: str) -> str:
        process = subprocess.Popen(
            ["ruff", "format", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=source_code)
        if process.returncode != 0:
            raise Exception(f"Ruff formatting failed: {stderr}")
        return stdout

    return "\n".join(
        [
            f"\\begin{{lstlisting}}[style={style}, caption={{{caption}}}, label={{code:{label}}}]",
            format_string_with_ruff(code),
            "\end{lstlisting}",
        ]
    )


def definition_to_code_string(obj, var_name: str):
    if isinstance(obj, list):
        lines = [f"{var_name} = ["]

        for item in obj:
            if not isinstance(item, str) or "\\" not in item:
                lines.append(f"{pprint.pformat(item, width=100, sort_dicts=False)},")
                continue

            escaped = item.replace('"', r"\"").replace("'", r"\'")
            lines.append(f'r"{escaped}", ')

        lines.append("]")
        return "\n".join(lines)

    value_str = pprint.pformat(obj, width=100, sort_dicts=False)
    return f"{var_name} = {value_str}"
