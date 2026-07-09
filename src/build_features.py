"""Build modeling datasets from cleaned sales data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_SALES_PATH,
    DAILY_BUSINESS_METRICS_PATH,
    LATEST_INVENTORY_SNAPSHOT_PATH,
    PRODUCT_DAILY_DEMAND_PATH,
    PRODUCT_STORE_DAILY_DEMAND_PATH,
)
from src.save_outputs import save_csv


def _read_cleaned(df: pd.DataFrame | None, input_path: str | Path) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(input_path)
    output = df.copy()
    output["calendar_date"] = pd.to_datetime(output["calendar_date"], errors="coerce")
    return output.dropna(subset=["calendar_date"])


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def build_daily_business_metrics(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("calendar_date", as_index=False)[["revenue", "gross_profit", "units"]]
        .sum()
        .rename(columns={"calendar_date": "date"})
    )

    if not daily.empty:
        full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = (
            daily.set_index("date")
            .reindex(full_dates)
            .fillna({"revenue": 0, "gross_profit": 0, "units": 0})
            .rename_axis("date")
            .reset_index()
        )

    save_csv(daily, DAILY_BUSINESS_METRICS_PATH)
    return daily


def build_product_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = _available_columns(
        df,
        [
            "calendar_date",
            "product_key",
            "product_id",
            "product_name",
            "product_category",
        ],
    )
    agg_map = {
        "units": "sum",
        "revenue": "sum",
        "gross_profit": "sum",
        "unit_margin": "mean",
        "product_margin": "mean",
    }
    agg_map = {column: agg for column, agg in agg_map.items() if column in df.columns}

    product_daily = (
        df.groupby(group_columns, as_index=False)
        .agg(agg_map)
        .rename(columns={"calendar_date": "date"})
    )
    save_csv(product_daily, PRODUCT_DAILY_DEMAND_PATH)
    return product_daily


def build_product_store_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = _available_columns(
        df,
        [
            "calendar_date",
            "store_key",
            "store_id",
            "store_name",
            "store_city",
            "store_location",
            "product_key",
            "product_id",
            "product_name",
            "product_category",
        ],
    )
    agg_map = {
        "units": "sum",
        "revenue": "sum",
        "gross_profit": "sum",
        "unit_margin": "mean",
        "product_margin": "mean",
    }
    agg_map = {column: agg for column, agg in agg_map.items() if column in df.columns}

    product_store_daily = (
        df.groupby(group_columns, as_index=False)
        .agg(agg_map)
        .rename(columns={"calendar_date": "date"})
    )
    save_csv(product_store_daily, PRODUCT_STORE_DAILY_DEMAND_PATH)
    return product_store_daily


def build_latest_inventory_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["calendar_date"]
    if "sales_key" in df.columns:
        sort_columns.append("sales_key")

    latest = (
        df.sort_values(sort_columns)
        .groupby(["store_key", "product_key"], as_index=False)
        .tail(1)
        .copy()
    )

    column_map = {
        "calendar_date": "latest_date",
        "stock_on_hand": "current_stock",
        "stock_status": "latest_stock_status",
    }
    latest = latest.rename(columns=column_map)

    desired_columns = _available_columns(
        latest,
        [
            "latest_date",
            "store_key",
            "store_id",
            "store_name",
            "store_city",
            "store_location",
            "product_key",
            "product_id",
            "product_name",
            "product_category",
            "current_stock",
            "latest_stock_status",
            "out_of_stock_flag",
            "restock_risk_flag",
            "is_out_of_stock",
            "is_restock_risk",
            "unit_margin",
            "product_margin",
        ],
    )
    latest = latest[desired_columns].sort_values(["store_key", "product_key"])
    save_csv(latest, LATEST_INVENTORY_SNAPSHOT_PATH)
    return latest


def build_all_features(
    df: pd.DataFrame | None = None,
    input_path: str | Path = CLEANED_SALES_PATH,
) -> dict[str, pd.DataFrame]:
    cleaned = _read_cleaned(df, input_path)
    return {
        "daily_business_metrics": build_daily_business_metrics(cleaned),
        "product_daily_demand": build_product_daily_demand(cleaned),
        "product_store_daily_demand": build_product_store_daily_demand(cleaned),
        "latest_inventory_snapshot": build_latest_inventory_snapshot(cleaned),
    }
