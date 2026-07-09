"""Project configuration for the Mexico Toy Store forecasting pipeline."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
FORECAST_OUTPUT_DIR = OUTPUTS_DIR / "forecasts"
RESTOCK_OUTPUT_DIR = OUTPUTS_DIR / "restock"
PROFIT_OPPORTUNITY_OUTPUT_DIR = OUTPUTS_DIR / "profit_opportunity"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
VISUALS_DIR = ROOT_DIR / "visuals"

RAW_SALES_PATH = RAW_DATA_DIR / "sales_joined_raw.csv"
CLEANED_SALES_PATH = PROCESSED_DATA_DIR / "sales_cleaned.csv"
DAILY_BUSINESS_METRICS_PATH = PROCESSED_DATA_DIR / "daily_business_metrics.csv"
PRODUCT_DAILY_DEMAND_PATH = PROCESSED_DATA_DIR / "product_daily_demand.csv"
PRODUCT_STORE_DAILY_DEMAND_PATH = PROCESSED_DATA_DIR / "product_store_daily_demand.csv"
LATEST_INVENTORY_SNAPSHOT_PATH = PROCESSED_DATA_DIR / "latest_inventory_snapshot.csv"

REVENUE_FORECAST_PATH = FORECAST_OUTPUT_DIR / "revenue_forecast_next_30_days.csv"
PROFIT_FORECAST_PATH = FORECAST_OUTPUT_DIR / "profit_forecast_next_30_days.csv"
PRODUCT_STORE_DEMAND_FORECAST_PATH = FORECAST_OUTPUT_DIR / "product_store_demand_forecast.csv"
PRODUCT_DEMAND_FORECAST_PATH = FORECAST_OUTPUT_DIR / "product_demand_forecast.csv"
RESTOCK_RECOMMENDATIONS_PATH = RESTOCK_OUTPUT_DIR / "restock_recommendations_next_7_days.csv"
TOP_PROFIT_PRODUCTS_PATH = PROFIT_OPPORTUNITY_OUTPUT_DIR / "top_profit_products_next_30_days.csv"
TOP_PROFIT_PRODUCTS_BY_STORE_PATH = (
    PROFIT_OPPORTUNITY_OUTPUT_DIR / "top_profit_products_by_store_next_30_days.csv"
)

REVENUE_EVALUATION_PATH = REPORTS_DIR / "revenue_model_evaluation.csv"
PROFIT_EVALUATION_PATH = REPORTS_DIR / "profit_model_evaluation.csv"
MODEL_EVALUATION_SUMMARY_PATH = REPORTS_DIR / "model_evaluation_summary.csv"
EVALUATION_RESULTS_MD_PATH = REPORTS_DIR / "evaluation_results.md"

DEFAULT_DATABASE = "TOY_SALES_DB"
DEFAULT_SCHEMA = "ANALYTICS_MARTS"
DEFAULT_WAREHOUSE = "ToyS_WH"

TABLES = {
    "date": "DIM_DATE",
    "inventory_status": "DIM_INVENTORY_STATUS",
    "product": "DIM_PRODUCT",
    "store": "DIM_STORE",
    "sales": "FCT_SALES",
}

FORECAST_HORIZONS = {
    "revenue_days": 30,
    "profit_days": 30,
    "restock_days": 7,
    "product_profit_days": 30,
}

RESTOCK_ASSUMPTIONS = {
    "supplier_lead_time_days": 7,
    "safety_stock_pct": 0.20,
    "large_days_of_inventory": 9999.0,
}

MIN_PRODUCT_STORE_RECORDS = 14
MIN_PRODUCT_RECORDS = 14
MIN_CATEGORY_RECORDS = 21
MOVING_AVERAGE_WINDOW = 7
SEASONAL_PERIODS = 7
MIN_SEASONAL_POINTS = SEASONAL_PERIODS * 2
EVALUATION_HOLDOUT_DAYS = 30


PROJECT_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FORECAST_OUTPUT_DIR,
    RESTOCK_OUTPUT_DIR,
    PROFIT_OPPORTUNITY_OUTPUT_DIR,
    ROOT_DIR / "notebooks",
    MODELS_DIR / "revenue",
    MODELS_DIR / "profit",
    MODELS_DIR / "demand",
    REPORTS_DIR,
    VISUALS_DIR / "revenue",
    VISUALS_DIR / "profit",
    VISUALS_DIR / "demand",
    VISUALS_DIR / "restock",
]


def ensure_project_directories() -> None:
    """Create all expected project directories if they do not already exist."""
    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def quote_identifier(identifier: str) -> str:
    """Safely quote a Snowflake identifier."""
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def qualified_table(table_name: str, database: str = DEFAULT_DATABASE, schema: str = DEFAULT_SCHEMA) -> str:
    """Return a fully qualified Snowflake table name."""
    return ".".join(
        [
            quote_identifier(database),
            quote_identifier(schema),
            quote_identifier(table_name),
        ]
    )
