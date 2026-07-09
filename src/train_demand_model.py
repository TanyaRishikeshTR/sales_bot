"""Forecast product and product-store demand for restock planning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_HORIZONS,
    MIN_CATEGORY_RECORDS,
    MIN_PRODUCT_RECORDS,
    MIN_PRODUCT_STORE_RECORDS,
    PRODUCT_DAILY_DEMAND_PATH,
    PRODUCT_DEMAND_FORECAST_PATH,
    PRODUCT_STORE_DAILY_DEMAND_PATH,
    PRODUCT_STORE_DEMAND_FORECAST_PATH,
)
from src.forecast_utils import as_daily_series, exponential_smoothing_forecast, moving_average_forecast
from src.save_outputs import save_csv


PRODUCT_COLUMNS = ["product_key", "product_id", "product_name", "product_category"]
STORE_COLUMNS = ["store_key", "store_id", "store_name", "store_city", "store_location"]
MARGIN_COLUMNS = ["unit_margin", "product_margin"]


def _metadata_row(df: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    row = df.sort_values("date").tail(1)
    return {column: row[column].iloc[0] if column in row.columns and not row.empty else np.nan for column in columns}


def _series_for_group(group: pd.DataFrame) -> pd.Series:
    return as_daily_series(group, "date", "units", fill_value=0)


def _forecast_units(series: pd.Series, horizon: int, prefer_exponential: bool) -> tuple[float, str]:
    if prefer_exponential and len(series) >= 14:
        result = exponential_smoothing_forecast(series, horizon)
    else:
        result = moving_average_forecast(series, horizon)
    return float(result.forecast.sum()), result.model_used


def _build_category_per_combo_forecasts(product_store_daily: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    forecasts: dict[str, dict[str, float | str]] = {}
    if "product_category" not in product_store_daily.columns:
        return forecasts

    combo_counts = (
        product_store_daily.groupby("product_category")[["store_key", "product_key"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .replace(0, 1)
    )

    for category, group in product_store_daily.groupby("product_category"):
        category_daily = group.groupby("date", as_index=False)["units"].sum()
        series = _series_for_group(category_daily)
        divisor = float(combo_counts.loc[category])
        forecast_7, method_7 = _forecast_units(series / divisor, FORECAST_HORIZONS["restock_days"], False)
        forecast_30, _ = _forecast_units(series / divisor, FORECAST_HORIZONS["product_profit_days"], False)
        forecasts[str(category)] = {
            "forecasted_units_next_7_days": forecast_7,
            "forecasted_units_next_30_days": forecast_30,
            "demand_model_used": f"category_average_per_product_store_{method_7}",
        }
    return forecasts


def build_product_level_demand_forecast(product_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    category_forecasts: dict[str, dict[str, float | str]] = {}

    if "product_category" in product_daily.columns:
        for category, group in product_daily.groupby("product_category"):
            category_daily = group.groupby("date", as_index=False)["units"].sum()
            product_count = max(1, group["product_key"].nunique())
            series = _series_for_group(category_daily) / product_count
            forecast_7, method_7 = _forecast_units(series, FORECAST_HORIZONS["restock_days"], False)
            forecast_30, _ = _forecast_units(series, FORECAST_HORIZONS["product_profit_days"], False)
            category_forecasts[str(category)] = {
                "forecasted_units_next_7_days": forecast_7,
                "forecasted_units_next_30_days": forecast_30,
                "demand_model_used": f"category_average_per_product_{method_7}",
            }

    for product_key, group in product_daily.groupby("product_key"):
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"], errors="coerce")
        metadata = _metadata_row(group, PRODUCT_COLUMNS + MARGIN_COLUMNS)
        series = _series_for_group(group)

        if group.shape[0] >= MIN_PRODUCT_RECORDS:
            forecast_7, method_7 = _forecast_units(series, FORECAST_HORIZONS["restock_days"], True)
            forecast_30, _ = _forecast_units(series, FORECAST_HORIZONS["product_profit_days"], True)
            model_used = f"product_level_{method_7}"
        else:
            category = str(metadata.get("product_category", ""))
            fallback = category_forecasts.get(category)
            if fallback and group.shape[0] < MIN_CATEGORY_RECORDS:
                forecast_7 = float(fallback["forecasted_units_next_7_days"])
                forecast_30 = float(fallback["forecasted_units_next_30_days"])
                model_used = str(fallback["demand_model_used"])
            else:
                forecast_7, method_7 = _forecast_units(series, FORECAST_HORIZONS["restock_days"], False)
                forecast_30, _ = _forecast_units(series, FORECAST_HORIZONS["product_profit_days"], False)
                model_used = f"sparse_product_{method_7}"

        rows.append(
            {
                **metadata,
                "product_key": product_key,
                "forecasted_units_next_7_days": max(0.0, forecast_7),
                "forecasted_units_next_30_days": max(0.0, forecast_30),
                "demand_model_used": model_used,
            }
        )

    output = pd.DataFrame(rows)
    return output.sort_values("forecasted_units_next_30_days", ascending=False)


def build_product_store_demand_forecast(
    product_store_daily: pd.DataFrame,
    product_forecast: pd.DataFrame,
) -> pd.DataFrame:
    category_fallbacks = _build_category_per_combo_forecasts(product_store_daily)
    active_stores_per_product = product_store_daily.groupby("product_key")["store_key"].nunique().to_dict()
    product_forecast_lookup = product_forecast.set_index("product_key").to_dict("index")
    rows: list[dict[str, object]] = []

    group_columns = ["store_key", "product_key"]
    for (store_key, product_key), group in product_store_daily.groupby(group_columns):
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"], errors="coerce")
        metadata = _metadata_row(group, STORE_COLUMNS + PRODUCT_COLUMNS + MARGIN_COLUMNS)
        series = _series_for_group(group)

        if group.shape[0] >= MIN_PRODUCT_STORE_RECORDS:
            forecast_7, method_7 = _forecast_units(series, FORECAST_HORIZONS["restock_days"], True)
            forecast_30, _ = _forecast_units(series, FORECAST_HORIZONS["product_profit_days"], True)
            model_used = f"product_store_{method_7}"
        elif product_key in product_forecast_lookup:
            product_row = product_forecast_lookup[product_key]
            divisor = max(1, active_stores_per_product.get(product_key, 1))
            forecast_7 = float(product_row["forecasted_units_next_7_days"]) / divisor
            forecast_30 = float(product_row["forecasted_units_next_30_days"]) / divisor
            model_used = "fallback_product_average_per_active_store"
        else:
            category = str(metadata.get("product_category", ""))
            fallback = category_fallbacks.get(category, {})
            forecast_7 = float(fallback.get("forecasted_units_next_7_days", 0.0))
            forecast_30 = float(fallback.get("forecasted_units_next_30_days", 0.0))
            model_used = str(fallback.get("demand_model_used", "fallback_zero_demand"))

        rows.append(
            {
                **metadata,
                "store_key": store_key,
                "product_key": product_key,
                "forecasted_units_next_7_days": max(0.0, forecast_7),
                "forecasted_units_next_30_days": max(0.0, forecast_30),
                "demand_model_used": model_used,
            }
        )

    output = pd.DataFrame(rows)
    sort_columns = [column for column in ["store_name", "product_name"] if column in output.columns]
    if sort_columns:
        output = output.sort_values(sort_columns)
    return output


def train_demand_model(
    product_store_path: str | Path = PRODUCT_STORE_DAILY_DEMAND_PATH,
    product_path: str | Path = PRODUCT_DAILY_DEMAND_PATH,
    product_store_output_path: str | Path = PRODUCT_STORE_DEMAND_FORECAST_PATH,
    product_output_path: str | Path = PRODUCT_DEMAND_FORECAST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_store_daily = pd.read_csv(product_store_path)
    product_daily = pd.read_csv(product_path)
    product_store_daily["date"] = pd.to_datetime(product_store_daily["date"], errors="coerce")
    product_daily["date"] = pd.to_datetime(product_daily["date"], errors="coerce")

    product_forecast = build_product_level_demand_forecast(product_daily)
    product_store_forecast = build_product_store_demand_forecast(product_store_daily, product_forecast)

    save_csv(product_forecast, product_output_path)
    save_csv(product_store_forecast, product_store_output_path)
    return product_store_forecast, product_forecast


if __name__ == "__main__":
    train_demand_model()
