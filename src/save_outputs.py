"""Small output helpers used across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def ensure_parent_dir(path: PathLike) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_csv(df: pd.DataFrame, path: PathLike, index: bool = False) -> Path:
    output_path = ensure_parent_dir(path)
    df.to_csv(output_path, index=index)
    return output_path


def save_markdown(content: str, path: PathLike) -> Path:
    output_path = ensure_parent_dir(path)
    output_path.write_text(content, encoding="utf-8")
    return output_path
