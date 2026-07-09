"""Cached CSV loaders for the Streamlit ML assistant."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "revenue_forecast": {
        "label": "Revenue forecast",
        "path": PROJECT_ROOT / "data" / "outputs" / "forecasts" / "revenue_forecast_next_30_days.csv",
    },
    "profit_forecast": {
        "label": "Gross profit forecast",
        "path": PROJECT_ROOT / "data" / "outputs" / "forecasts" / "profit_forecast_next_30_days.csv",
    },
    "product_store_demand": {
        "label": "Product-store demand forecast",
        "path": PROJECT_ROOT / "data" / "outputs" / "forecasts" / "product_store_demand_forecast.csv",
    },
    "product_demand": {
        "label": "Product demand forecast",
        "path": PROJECT_ROOT / "data" / "outputs" / "forecasts" / "product_demand_forecast.csv",
    },
    "restock": {
        "label": "Restock recommendations",
        "path": PROJECT_ROOT
        / "data"
        / "outputs"
        / "restock"
        / "restock_recommendations_next_7_days.csv",
    },
    "top_profit_products": {
        "label": "Top profit products",
        "path": PROJECT_ROOT
        / "data"
        / "outputs"
        / "profit_opportunity"
        / "top_profit_products_next_30_days.csv",
    },
    "top_profit_products_by_store": {
        "label": "Top profit products by store",
        "path": PROJECT_ROOT
        / "data"
        / "outputs"
        / "profit_opportunity"
        / "top_profit_products_by_store_next_30_days.csv",
    },
    "model_evaluation": {
        "label": "Model evaluation",
        "path": PROJECT_ROOT / "reports" / "model_evaluation_summary.csv",
    },
}

DISABLED_DATASET_KEYS: set[str] = {"product_demand", "product_store_demand", "restock"}

DISABLED_MODULES: list[str] = [
    "Demand forecast",
    "Product-store demand forecast",
    "Restock recommendations",
]

HISTORICAL_SALES_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "sales_cleaned.csv.gz",  # Main compressed file (use this for Streamlit Cloud)
    PROJECT_ROOT / "data" / "processed" / "sales_cleaned.csv",     # Fallback local uncompressed file
    PROJECT_ROOT / "data" / "raw" / "sales_joined_raw.csv.gz",     # Compressed raw fallback
    PROJECT_ROOT / "data" / "raw" / "sales_joined_raw.csv",        # Fallback raw file
]


def to_snake_case(name: str) -> str:
    """Normalize a column name to lowercase snake_case."""
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [to_snake_case(column) for column in output.columns]
    output = drop_auto_index_columns(output)
    return output


def drop_auto_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CSV row-index columns that should never be shown to users."""
    if df.empty:
        return df.copy()

    output = df.copy()
    drop_columns = [
        column
        for column in output.columns
        if str(column).strip().lower().startswith("unnamed")
        or str(column).strip().lower() in {"index", "level_0", "row_index"}
    ]
    if drop_columns:
        output = output.drop(columns=drop_columns, errors="ignore")
    return output


@st.cache_data(show_spinner=False)
def load_csv_safely(path_text: str) -> tuple[pd.DataFrame, str | None]:
    """Load a CSV without crashing the app when files are missing or malformed."""
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame(), f"Missing file: {path}"

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {path.name}: {exc}"

    if df.empty:
        return standardize_columns(df), f"File is empty: {path.name}"

    return standardize_columns(df), None


def load_csv_safe(path: str | Path) -> pd.DataFrame:
    """Compatibility helper that returns only the dataframe."""
    df, _ = load_csv_safely(str(path))
    return df


def _load_dataset(key: str) -> tuple[pd.DataFrame, str | None]:
    return load_csv_safely(str(DATASETS[key]["path"]))


def load_revenue_forecast() -> tuple[pd.DataFrame, str | None]:
    return _load_dataset("revenue_forecast")


def load_profit_forecast() -> tuple[pd.DataFrame, str | None]:
    return _load_dataset("profit_forecast")


def load_restock_recommendations() -> tuple[pd.DataFrame, str | None]:
    return _load_dataset("restock")


def load_top_profit_products() -> tuple[pd.DataFrame, str | None]:
    return _load_dataset("top_profit_products")


def load_product_demand_forecast() -> tuple[pd.DataFrame, str | None]:
    return _load_dataset("product_demand")


def load_model_evaluation() -> tuple[pd.DataFrame, str | None]:
    return _load_dataset("model_evaluation")


def load_historical_sales() -> tuple[pd.DataFrame, str | None, Path | None]:
    """Load processed historical sales, falling back to the raw joined extract."""
    missing_paths = []
    for path in HISTORICAL_SALES_PATHS:
        df, warning = load_csv_safely(str(path))
        if warning is None and not df.empty:
            return df, None, path
        missing_paths.append(str(path))

    return (
        pd.DataFrame(),
        "Missing historical sales file. Checked: " + "; ".join(missing_paths),
        None,
    )


def load_all_data() -> dict[str, dict[str, object]]:
    """Load every app dataset and include status metadata for the sidebar."""
    loaded: dict[str, dict[str, object]] = {}
    for key, meta in DATASETS.items():
        df, warning = _load_dataset(key)
        disabled = key in DISABLED_DATASET_KEYS
        loaded[key] = {
            "label": meta["label"],
            "path": meta["path"],
            "df": df,
            "warning": warning,
            "available": warning is None and not df.empty,
            "disabled": disabled,
            "display_status": "loaded but disabled" if disabled and warning is None and not df.empty else None,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
        }
    historical_df, historical_warning, historical_path = load_historical_sales()
    loaded["historical_sales"] = {
        "label": "Historical sales",
        "path": historical_path or HISTORICAL_SALES_PATHS[0],
        "df": historical_df,
        "warning": historical_warning,
        "available": historical_warning is None and not historical_df.empty,
        "rows": int(len(historical_df)),
        "columns": int(len(historical_df.columns)),
        "column_names": list(historical_df.columns),
    }
    return loaded
