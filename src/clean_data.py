"""Clean raw sales extract data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEANED_SALES_PATH, RAW_SALES_PATH
from src.save_outputs import save_csv


NUMERIC_COLUMNS = [
    "year",
    "quarter",
    "month_number",
    "week_of_year",
    "product_cost",
    "product_price",
    "product_margin",
    "product_margin_pct",
    "units",
    "unit_price",
    "unit_cost",
    "unit_margin",
    "revenue",
    "total_cost",
    "gross_profit",
    "gross_margin_pct",
    "stock_on_hand",
    "inventory_cost_value",
    "inventory_retail_value",
    "potential_gross_profit",
    "out_of_stock_flag",
    "restock_risk_flag",
    "is_out_of_stock",
    "is_restock_risk",
]

REQUIRED_COLUMNS = ["calendar_date", "product_key", "store_key", "units"]


def to_snake_case(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [to_snake_case(column) for column in output.columns]
    return output


def clean_sales_data(
    df: pd.DataFrame | None = None,
    input_path: str | Path = RAW_SALES_PATH,
    output_path: str | Path = CLEANED_SALES_PATH,
) -> pd.DataFrame:
    """Clean the raw joined sales extract and save a processed CSV."""
    if df is None:
        df = pd.read_csv(input_path)

    cleaned = standardize_columns(df)

    if "calendar_date" not in cleaned.columns:
        raise ValueError("Expected column calendar_date in raw sales data.")

    cleaned["calendar_date"] = pd.to_datetime(cleaned["calendar_date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    missing_required = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)

    measure_columns = [
        "units",
        "revenue",
        "gross_profit",
        "stock_on_hand",
        "unit_margin",
        "product_margin",
        "out_of_stock_flag",
        "restock_risk_flag",
        "is_out_of_stock",
        "is_restock_risk",
    ]
    for column in measure_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0)

    text_columns = cleaned.select_dtypes(include=["object"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].fillna("unknown")

    if "sales_key" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["sales_key"])
    elif "sale_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["sale_id"])
    else:
        cleaned = cleaned.drop_duplicates()

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0)

    cleaned = cleaned.sort_values(["calendar_date", "store_key", "product_key"]).reset_index(drop=True)
    save_csv(cleaned, output_path)
    return cleaned
