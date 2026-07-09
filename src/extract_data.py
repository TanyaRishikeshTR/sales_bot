"""Extract joined sales data from Snowflake into a local raw CSV."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    DEFAULT_DATABASE,
    DEFAULT_SCHEMA,
    RAW_SALES_PATH,
    ROOT_DIR,
    TABLES,
    qualified_table,
)
from src.save_outputs import save_csv
from src.snowflake_connection import get_snowflake_connection


def build_sales_extract_query() -> str:
    """Build the read-only SELECT query used for extraction."""
    load_dotenv(ROOT_DIR / ".env")
    database = os.getenv("SF_DATABASE", DEFAULT_DATABASE)
    schema = os.getenv("SF_SCHEMA", DEFAULT_SCHEMA)

    sales = qualified_table(TABLES["sales"], database, schema)
    date = qualified_table(TABLES["date"], database, schema)
    product = qualified_table(TABLES["product"], database, schema)
    store = qualified_table(TABLES["store"], database, schema)
    inventory = qualified_table(TABLES["inventory_status"], database, schema)

    return f"""
SELECT
    f.SALES_KEY,
    f.SALE_ID,
    f.DATE_KEY,
    d.CALENDAR_DATE,
    d.YEAR,
    d.QUARTER,
    d.MONTH_NUMBER,
    d.MONTH_NAME,
    d.WEEK_OF_YEAR,
    d.DAY_NAME,
    d.IS_WEEKEND,
    d.DAY_TYPE,
    f.PRODUCT_KEY,
    p.PRODUCT_ID,
    p.PRODUCT_NAME,
    p.PRODUCT_CATEGORY,
    p.PRODUCT_COST,
    p.PRODUCT_PRICE,
    p.PRODUCT_MARGIN,
    p.PRODUCT_MARGIN_PCT,
    p.PRICE_BAND,
    p.MARGIN_BAND,
    f.STORE_KEY,
    s.STORE_ID,
    s.STORE_NAME,
    s.STORE_CITY,
    s.STORE_LOCATION,
    s.STORE_OPEN_DATE,
    s.STORE_MATURITY_BAND,
    f.INVENTORY_STATUS_KEY,
    i.STOCK_STATUS,
    i.IS_OUT_OF_STOCK,
    i.IS_RESTOCK_RISK,
    i.RESTOCK_PRIORITY,
    f.UNITS,
    f.UNIT_PRICE,
    f.UNIT_COST,
    f.UNIT_MARGIN,
    f.REVENUE,
    f.TOTAL_COST,
    f.GROSS_PROFIT,
    f.GROSS_MARGIN_PCT,
    f.STOCK_ON_HAND,
    f.INVENTORY_COST_VALUE,
    f.INVENTORY_RETAIL_VALUE,
    f.POTENTIAL_GROSS_PROFIT,
    f.OUT_OF_STOCK_FLAG,
    f.RESTOCK_RISK_FLAG
FROM {sales} f
INNER JOIN {date} d
    ON f.DATE_KEY = d.DATE_KEY
INNER JOIN {product} p
    ON f.PRODUCT_KEY = p.PRODUCT_KEY
INNER JOIN {store} s
    ON f.STORE_KEY = s.STORE_KEY
LEFT JOIN {inventory} i
    ON f.INVENTORY_STATUS_KEY = i.INVENTORY_STATUS_KEY
"""


def extract_sales_data(connection: Any | None = None, output_path=RAW_SALES_PATH) -> pd.DataFrame:
    """Read joined sales data from Snowflake and save it locally."""
    owns_connection = connection is None
    if connection is None:
        connection = get_snowflake_connection()

    try:
        df = pd.read_sql(build_sales_extract_query(), connection)
    finally:
        if owns_connection:
            connection.close()

    save_csv(df, output_path)
    return df
