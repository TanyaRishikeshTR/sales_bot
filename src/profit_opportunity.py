"""Forecasted gross profit opportunity rankings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    LATEST_INVENTORY_SNAPSHOT_PATH,
    PRODUCT_DEMAND_FORECAST_PATH,
    PRODUCT_STORE_DEMAND_FORECAST_PATH,
    TOP_PROFIT_PRODUCTS_BY_STORE_PATH,
    TOP_PROFIT_PRODUCTS_PATH,
)
from src.save_outputs import save_csv


def _margin_series(df: pd.DataFrame) -> pd.Series:
    margin = pd.Series(0.0, index=df.index)
    for column in ["unit_margin", "product_margin", "unit_margin_inventory", "product_margin_inventory"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            margin = margin.where(margin > 0, values)
    return margin.fillna(0)


def generate_profit_opportunity(
    product_forecast_path: str | Path = PRODUCT_DEMAND_FORECAST_PATH,
    product_store_forecast_path: str | Path = PRODUCT_STORE_DEMAND_FORECAST_PATH,
    inventory_path: str | Path = LATEST_INVENTORY_SNAPSHOT_PATH,
    product_output_path: str | Path = TOP_PROFIT_PRODUCTS_PATH,
    product_store_output_path: str | Path = TOP_PROFIT_PRODUCTS_BY_STORE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_forecast = pd.read_csv(product_forecast_path)
    product_store_forecast = pd.read_csv(product_store_forecast_path)
    inventory = pd.read_csv(inventory_path)

    inventory_product = (
        inventory.groupby("product_key", as_index=False)
        .agg(
            current_stock=("current_stock", "sum"),
            out_of_stock_flag=("out_of_stock_flag", "max"),
            restock_risk_flag=("restock_risk_flag", "max"),
            unit_margin_inventory=("unit_margin", "mean"),
            product_margin_inventory=("product_margin", "mean"),
        )
    )

    product_output = product_forecast.merge(inventory_product, on="product_key", how="left")
    product_output["unit_margin"] = _margin_series(product_output)
    product_output["forecasted_gross_profit_next_30_days"] = (
        pd.to_numeric(product_output["forecasted_units_next_30_days"], errors="coerce").fillna(0)
        * product_output["unit_margin"]
    )
    product_output = product_output.sort_values(
        "forecasted_gross_profit_next_30_days", ascending=False
    ).reset_index(drop=True)
    product_output.insert(0, "profit_rank", product_output.index + 1)

    inventory_store_columns = [
        "store_key",
        "product_key",
        "current_stock",
        "out_of_stock_flag",
        "restock_risk_flag",
        "latest_stock_status",
        "unit_margin",
        "product_margin",
    ]
    inventory_store_columns = [column for column in inventory_store_columns if column in inventory.columns]
    product_store_output = product_store_forecast.merge(
        inventory[inventory_store_columns],
        on=["store_key", "product_key"],
        how="left",
        suffixes=("", "_inventory"),
    )
    product_store_output["unit_margin"] = _margin_series(product_store_output)
    product_store_output["forecasted_store_product_profit_next_30_days"] = (
        pd.to_numeric(product_store_output["forecasted_units_next_30_days"], errors="coerce").fillna(0)
        * product_store_output["unit_margin"]
    )
    product_store_output = product_store_output.sort_values(
        "forecasted_store_product_profit_next_30_days", ascending=False
    ).reset_index(drop=True)
    product_store_output.insert(0, "overall_profit_rank", product_store_output.index + 1)
    product_store_output["store_profit_rank"] = (
        product_store_output.groupby("store_key")["forecasted_store_product_profit_next_30_days"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    save_csv(product_output, product_output_path)
    save_csv(product_store_output, product_store_output_path)
    return product_output, product_store_output


if __name__ == "__main__":
    generate_profit_opportunity()
