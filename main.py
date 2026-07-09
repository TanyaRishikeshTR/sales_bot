"""Run the Mexico Toy Store sales forecasting pipeline end to end."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from src.build_features import build_all_features
from src.clean_data import clean_sales_data
from src.config import (
    CLEANED_SALES_PATH,
    LATEST_INVENTORY_SNAPSHOT_PATH,
    MODEL_EVALUATION_SUMMARY_PATH,
    PRODUCT_DEMAND_FORECAST_PATH,
    PRODUCT_STORE_DEMAND_FORECAST_PATH,
    PROFIT_FORECAST_PATH,
    RAW_SALES_PATH,
    RESTOCK_RECOMMENDATIONS_PATH,
    REVENUE_FORECAST_PATH,
    TOP_PROFIT_PRODUCTS_BY_STORE_PATH,
    TOP_PROFIT_PRODUCTS_PATH,
    ensure_project_directories,
)
from src.evaluate_models import evaluate_all_models
from src.extract_data import extract_sales_data
from src.profit_opportunity import generate_profit_opportunity
from src.restock_rules import generate_restock_recommendations
from src.train_demand_model import train_demand_model
from src.train_profit_model import train_profit_model
from src.train_revenue_model import train_revenue_model


FINAL_OUTPUTS = [
    REVENUE_FORECAST_PATH,
    PROFIT_FORECAST_PATH,
    PRODUCT_STORE_DEMAND_FORECAST_PATH,
    PRODUCT_DEMAND_FORECAST_PATH,
    RESTOCK_RECOMMENDATIONS_PATH,
    TOP_PROFIT_PRODUCTS_PATH,
    TOP_PROFIT_PRODUCTS_BY_STORE_PATH,
    MODEL_EVALUATION_SUMMARY_PATH,
]


def _extract_or_reuse_raw(skip_extract: bool = False) -> pd.DataFrame | None:
    if skip_extract:
        print("Skipping Snowflake extract by request.")
        return None

    try:
        print("Extracting read-only sales data from Snowflake...")
        return extract_sales_data()
    except Exception as exc:
        print("Snowflake extraction failed.")
        print(f"Reason: {exc}")
        if RAW_SALES_PATH.exists():
            print(f"Continuing with existing local raw file: {RAW_SALES_PATH}")
            return None
        print("No local raw CSV is available, so the pipeline cannot continue.")
        print("Traceback for troubleshooting:")
        traceback.print_exc()
        raise


def run_pipeline(skip_extract: bool = False) -> list[Path]:
    ensure_project_directories()

    raw_df = _extract_or_reuse_raw(skip_extract=skip_extract)
    cleaned = clean_sales_data(raw_df if raw_df is not None else None)
    build_all_features(cleaned)

    train_revenue_model()
    train_profit_model()
    train_demand_model()
    generate_restock_recommendations()
    generate_profit_opportunity()
    evaluate_all_models()

    print("\nPipeline complete. Final outputs:")
    for path in FINAL_OUTPUTS:
        print(f"- {path}")
    return FINAL_OUTPUTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mexico Toy Store ML forecasting pipeline.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Use data/raw/sales_joined_raw.csv instead of connecting to Snowflake.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(skip_extract=args.skip_extract)
