"""Rule-based reorder point recommendations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LATEST_INVENTORY_SNAPSHOT_PATH,
    PRODUCT_STORE_DEMAND_FORECAST_PATH,
    RESTOCK_ASSUMPTIONS,
    RESTOCK_RECOMMENDATIONS_PATH,
)
from src.save_outputs import save_csv


def _coalesce_columns(
    df: pd.DataFrame,
    candidates: list[str],
    default=0,
    treat_zero_as_missing: bool = False,
) -> pd.Series:
    result = pd.Series(default, index=df.index)
    for column in candidates:
        if column in df.columns:
            keep_existing = result.notna()
            if not pd.isna(default):
                keep_existing = keep_existing & (result != default)
            if treat_zero_as_missing:
                keep_existing = keep_existing & (result != 0)
            result = result.where(keep_existing, df[column])
    return result


def _coalesce_dimension_columns(df: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in base_columns:
        if column in output.columns:
            continue
        inventory_column = f"{column}_inventory"
        forecast_column = f"{column}_forecast"
        if inventory_column in output.columns and forecast_column in output.columns:
            output[column] = output[inventory_column].combine_first(output[forecast_column])
        elif inventory_column in output.columns:
            output[column] = output[inventory_column]
        elif forecast_column in output.columns:
            output[column] = output[forecast_column]
    return output


def _priority_bands(df: pd.DataFrame) -> pd.Series:
    bands = pd.Series("Low", index=df.index)
    restock_scores = df.loc[df["needs_restock"], "priority_score"]
    restock_scores = restock_scores[restock_scores > 0]
    if restock_scores.empty:
        bands.loc[df["needs_restock"]] = "Medium"
        return bands

    high_threshold = restock_scores.quantile(0.67)
    medium_threshold = restock_scores.quantile(0.33)
    bands.loc[df["needs_restock"] & (df["priority_score"] >= high_threshold)] = "High"
    bands.loc[
        df["needs_restock"]
        & (df["priority_score"] < high_threshold)
        & (df["priority_score"] >= medium_threshold)
    ] = "Medium"
    bands.loc[df["needs_restock"] & (df["priority_score"] < medium_threshold)] = "Low"
    return bands


def generate_restock_recommendations(
    inventory_path: str | Path = LATEST_INVENTORY_SNAPSHOT_PATH,
    demand_forecast_path: str | Path = PRODUCT_STORE_DEMAND_FORECAST_PATH,
    output_path: str | Path = RESTOCK_RECOMMENDATIONS_PATH,
) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path)
    demand = pd.read_csv(demand_forecast_path)

    merged = inventory.merge(
        demand,
        on=["store_key", "product_key"],
        how="left",
        suffixes=("_inventory", "_forecast"),
    )
    merged = _coalesce_dimension_columns(
        merged,
        [
            "store_id",
            "store_name",
            "store_city",
            "store_location",
            "product_id",
            "product_name",
            "product_category",
        ],
    )

    merged["forecasted_units_next_7_days"] = pd.to_numeric(
        merged.get("forecasted_units_next_7_days", 0), errors="coerce"
    ).fillna(0)
    merged["current_stock"] = pd.to_numeric(merged.get("current_stock", 0), errors="coerce").fillna(0)

    safety_stock_pct = RESTOCK_ASSUMPTIONS["safety_stock_pct"]
    merged["forecasted_demand_during_lead_time"] = merged["forecasted_units_next_7_days"]
    merged["safety_stock"] = merged["forecasted_units_next_7_days"] * safety_stock_pct
    merged["reorder_point"] = merged["forecasted_demand_during_lead_time"] + merged["safety_stock"]
    merged["needs_restock"] = merged["current_stock"] <= merged["reorder_point"]
    merged["recommended_restock_quantity"] = np.ceil(
        np.maximum(0, merged["reorder_point"] - merged["current_stock"])
    ).astype(int)

    average_daily_forecast = merged["forecasted_units_next_7_days"] / RESTOCK_ASSUMPTIONS[
        "supplier_lead_time_days"
    ]
    merged["days_of_inventory_remaining"] = np.where(
        average_daily_forecast > 0,
        merged["current_stock"] / average_daily_forecast,
        RESTOCK_ASSUMPTIONS["large_days_of_inventory"],
    )

    merged["unit_margin_for_priority"] = _coalesce_columns(
        merged,
        [
            "unit_margin_inventory",
            "unit_margin_forecast",
            "unit_margin",
            "product_margin_inventory",
            "product_margin_forecast",
            "product_margin",
        ],
        default=np.nan,
        treat_zero_as_missing=True,
    )
    shortage = np.maximum(0, merged["reorder_point"] - merged["current_stock"])
    margin = pd.to_numeric(merged["unit_margin_for_priority"], errors="coerce")
    merged["priority_score"] = np.where(margin.fillna(0) > 0, shortage * margin.fillna(0), shortage)
    merged["priority_band"] = _priority_bands(merged)

    display_columns = [
        "store_name",
        "store_city",
        "store_location",
        "product_name",
        "product_category",
        "current_stock",
        "forecasted_units_next_7_days",
        "safety_stock",
        "reorder_point",
        "needs_restock",
        "recommended_restock_quantity",
        "days_of_inventory_remaining",
        "unit_margin_for_priority",
        "priority_score",
        "priority_band",
        "latest_stock_status",
        "out_of_stock_flag",
        "restock_risk_flag",
        "store_key",
        "product_key",
        "product_id",
        "store_id",
        "demand_model_used",
    ]
    display_columns = [column for column in display_columns if column in merged.columns]
    output = merged[display_columns].rename(columns={"unit_margin_for_priority": "unit_margin"})
    output = output.sort_values(["needs_restock", "priority_score"], ascending=[False, False])

    save_csv(output, output_path)
    return output


if __name__ == "__main__":
    generate_restock_recommendations()
