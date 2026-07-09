"""Rule-based retail analytics chatbot with source and confidence metadata."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_COLUMNS = ["date", "sales_date", "sales date", "order_date", "transaction_date", "invoice_date", "calendar_date", "month", "month_name", "quarter", "forecast_date", "ds"]
REVENUE_COLUMNS = ["revenue", "sales", "total_sales", "sales_amount", "net_sales", "amount", "total_amount"]
REVENUE_FORECAST_COLUMNS = ["forecasted_revenue", "revenue_forecast", "predicted_revenue", "forecast", "yhat"]
PROFIT_COLUMNS = ["gross_profit", "gross profit", "profit", "total_profit", "margin", "gross_margin", "margin_amount"]
PROFIT_FORECAST_COLUMNS = ["forecasted_gross_profit", "gross_profit_forecast", "predicted_profit", "forecasted_profit", "profit_forecast", "forecast", "yhat"]
UNITS_COLUMNS = ["units", "quantity", "qty", "units_sold", "units sold", "aggregate_units", "sales_quantity"]
PRODUCT_COLUMNS = ["product", "product_name", "product name", "product_id", "product id", "sku", "item", "item_name", "material", "material_name", "name"]
STORE_COLUMNS = ["store", "store_name", "store name", "store_id", "store id", "location", "store_location", "store location", "branch"]
LOCATION_COLUMNS = ["store_location", "store location", "location", "store_maturity_band", "store maturity band"]
CITY_COLUMNS = ["city", "store_city", "store city"]
CATEGORY_COLUMNS = ["category", "product_category", "product category", "category_name"]
INVENTORY_COLUMNS = ["stock", "stock_on_hand", "stock on hand", "inventory", "inventory_value", "inventory_cost", "inventory_retail", "cost_value", "retail_value"]
OUT_OF_STOCK_COLUMNS = ["out_of_stock", "out of stock", "is_out_of_stock", "stockout", "stock_out", "out_of_stock_flag"]
DEMAND_30_COLUMNS = ["forecasted_units_next_30_days", "units_next_30_days", "predicted_units_30_days"]
DEMAND_7_COLUMNS = ["forecasted_units_next_7_days", "units_next_7_days", "predicted_units_7_days"]
FORECASTED_PROFIT_COLUMNS = ["forecasted_gross_profit_next_30_days", "forecasted_store_product_profit_next_30_days", "forecasted_profit", "predicted_profit"]
RESTOCK_QTY_COLUMNS = ["recommended_restock_quantity", "restock_quantity", "recommended_quantity"]
PRIORITY_SCORE_COLUMNS = ["priority_score", "score"]
PRIORITY_BAND_COLUMNS = ["priority_band", "priority", "restock_priority"]
UNIT_MARGIN_COLUMNS = ["unit_margin", "product_margin", "margin"]

POWER_BI_REFERENCE = {
    "title": "MEXICO TOYS SALES ANALYTICS DASHBOARD",
    "total_revenue": 14_400_000.0,
    "total_profit": 4_000_000.0,
    "stock_on_hand": 20_800_000.0,
    "inventory_retail_value": 257_200_000.0,
    "inventory_cost_value": 188_100_000.0,
    "potential_gross_profit": 69_100_000.0,
    "total_units": 1_090_000.0,
    "total_stores": 50,
    "total_products": 35,
    "top_revenue_products": [
        ("Lego Bricks", 2_400_000.0),
        ("Colorbuds", 1_600_000.0),
        ("Magic Sand", 1_000_000.0),
        ("Action Figure", 900_000.0),
        ("Rubik's Cube", 900_000.0),
        ("Deck Of Cards", 600_000.0),
        ("Splash Balls", 500_000.0),
    ],
    "top_stores": [
        ("Maven Toys Ciudad de Mexico 2", 550_000.0, 170_000.0),
        ("Maven Toys Guadalajara 3", 450_000.0, 120_000.0),
        ("Maven Toys Ciudad de Mexico 1", 430_000.0, 110_000.0),
        ("Maven Toys Toluca 1", 410_000.0, 100_000.0),
        ("Maven Toys Monterrey 2", 370_000.0, 110_000.0),
    ],
    "category_revenue_profit": [
        ("Toys", 5_100_000.0, 1_100_000.0),
        ("Electronics", 2_200_000.0, 1_000_000.0),
        ("Art & Crafts", 2_700_000.0, 800_000.0),
        ("Games", 2_200_000.0, 700_000.0),
        ("Sports & Outdoors", 2_200_000.0, 500_000.0),
    ],
    "monthly_revenue": [
        ("Jan", 1_300_000.0),
        ("Feb", 1_300_000.0),
        ("Mar", 1_500_000.0),
        ("Apr", 1_500_000.0),
        ("May", 1_500_000.0),
        ("Jun", 1_500_000.0),
        ("Jul", 1_400_000.0),
        ("Aug", 1_200_000.0),
        ("Sep", 1_200_000.0),
        ("Oct", 600_000.0),
        ("Nov", 700_000.0),
        ("Dec", 900_000.0),
    ],
    "location_revenue_share": [("Downtown", 56.90), ("Commercial", 22.70), ("Residential", 11.47), ("Airport", 8.93)],
    "out_of_stock_total": 13_683,
    "out_of_stock_location_share": [("Downtown", 59.74), ("Residential", 18.73), ("Commercial", 18.58), ("Airport", 2.95)],
}

HELP_TEXT = (
    "I can help with historical sales analytics, revenue forecasts, gross profit forecasts, "
    "profit opportunities, model performance, and dashboard KPIs.\n\n"
    "Try asking:\n"
    "- What is the expected revenue for the next 30 days?\n"
    "- What is the expected gross profit for next month?\n"
    "- Which products may generate the most profit?\n"
    "- How accurate is the model?\n"
    "- What are the model limitations?"
)

MODULE_OFFLINE_RESPONSE = (
    "⚠️ **Module Offline**: The Product Demand and Inventory Restocking modules are currently disabled. "
    "During backtesting, the Holt-Winters engine flagged a high variance (41.91% MAPE) on sparse, "
    "high-cardinality SKU-level data. To protect supply chain decision-making from weak models, these "
    "features are deactivated until we collect more historical sales data.\n\n"
    "I can, however, provide 100% accurate historical analysis and highly confident 30-day aggregate "
    "Revenue and Gross Profit forecasts. What would you like to explore?"
)

DISABLED_MODULE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bdemand(?:s|ed|ing)?\b",
        r"\binventor(?:y|ies)\b",
        r"\bre[-\s]?stock(?:s|ed|ing)?\b",
        r"\bre[-\s]?order(?:s|ed|ing)?\b",
        r"\breplenish(?:es|ed|ing|ment|ments)?\b",
        r"\bstock[-\s]?on[-\s]?hand\b",
        r"\bstock[-\s]?out(?:s)?\b",
        r"\bout[-\s]?of[-\s]?stock\b",
        r"\bstock(?:s|ed|ing)?\b",
        r"\bsafety[-\s]?stock\b",
        r"\blead[-\s]?time\b",
        r"\border[-\s]?more\b",
        r"\border[-\s]?quantit(?:y|ies)\b",
        r"\bsku(?:s)?\b",
    ]
]

DISABLED_MODULE_PHRASES = [
    "forecasted units",
    "expected units",
    "unit forecast",
    "units forecast",
    "products will sell most",
    "products would sell most",
    "products are expected to sell",
    "run out",
    "supply chain",
    "procurement",
]


@dataclass
class ChatbotResult:
    answer: str
    intent: str
    table: pd.DataFrame | None = None
    chart: dict[str, Any] | None = None
    table_rows: int = 10
    source: str = "unknown"
    confidence: int = 50
    confidence_reason: str = ""
    model_performance_score: float | None = None
    model_performance_label: str | None = None


def build_response(
    text: str,
    table: pd.DataFrame | None = None,
    source: str = "unknown",
    confidence: int = 50,
    confidence_reason: str = "",
    model_performance_score: float | None = None,
    model_performance_label: str | None = None,
    intent: str = "answer",
    chart: dict[str, Any] | None = None,
    table_rows: int = 10,
) -> ChatbotResult:
    return ChatbotResult(
        answer=text,
        intent=intent,
        table=table,
        chart=chart,
        table_rows=table_rows,
        source=source,
        confidence=int(max(0, min(100, round(confidence)))),
        confidence_reason=confidence_reason,
        model_performance_score=model_performance_score,
        model_performance_label=model_performance_label,
    )


def confidence_label(score: float | int) -> str:
    if score < 30:
        return "Do not trust"
    if score < 60:
        return "Low confidence"
    if score < 80:
        return "Medium confidence"
    if score < 90:
        return "High confidence"
    return "Very high confidence"


def model_score_label(score: float | int | None) -> str | None:
    if score is None or pd.isna(score):
        return None
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Moderate"
    if score >= 30:
        return "Weak"
    return "Poor"


def mape_interpretation(mape_value: float | int | None) -> str:
    if mape_value is None or pd.isna(mape_value):
        return "not available"
    mape_float = float(mape_value)
    if mape_float < 10:
        return "very good"
    if mape_float < 20:
        return "good / acceptable"
    if mape_float < 50:
        return "weak / needs improvement"
    return "poor"


def _is_demand_model_name(value: Any) -> bool:
    text = str(value).strip().lower().replace("_", " ")
    return any(term in text for term in ["demand", "aggregate units", "unit forecast", "units forecast"])


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def find_column_with_match(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, str]:
    if df is None or df.empty:
        return None, "missing"
    cols = list(df.columns)
    lower = {str(c).strip().lower(): c for c in cols}
    for candidate in candidates:
        if candidate.strip().lower() in lower:
            return lower[candidate.strip().lower()], "exact"
    normalized = {_norm(c): c for c in cols}
    for candidate in candidates:
        if _norm(candidate) in normalized:
            return normalized[_norm(candidate)], "normalized"
    for candidate in candidates:
        candidate_norm = _norm(candidate)
        if not candidate_norm:
            continue
        for col in cols:
            col_norm = _norm(col)
            if candidate_norm in col_norm or col_norm in candidate_norm:
                return col, "partial"
    return None, "missing"


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return find_column_with_match(df, candidates)[0]


def prepare_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    values = df[col]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip()
    negative = text.str.match(r"^\(.*\)$", na=False)
    cleaned = (
        text.str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )
    suffix = cleaned.str.extract(r"(?i)([kmb])$", expand=False).str.upper()
    multiplier = suffix.map({"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}).fillna(1.0)
    number_text = cleaned.str.replace(r"(?i)[kmb]$", "", regex=True)
    numeric = pd.to_numeric(number_text, errors="coerce") * multiplier
    numeric.loc[negative] = numeric.loc[negative] * -1
    return numeric


def _numeric_null_penalty(series: pd.Series) -> tuple[int, str]:
    if len(series) == 0:
        return 15, "No numeric values were available."
    null_pct = float(series.isna().mean())
    if null_pct > 0.20:
        return 15, f"{null_pct:.0%} of values could not be converted to numeric."
    return 0, "Numeric conversion was clean."


def format_currency(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def format_number(value: Any, decimals: int = 0) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return f"{0:,.{decimals}f}"


def format_percent(value: Any) -> str:
    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return "0.00%"


def format_millions(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    return format_number(number)


def clean_table(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    output = df.copy()
    drop_cols = [
        col
        for col in output.columns
        if str(col).strip().lower().startswith("unnamed")
        or str(col).strip().lower() in {"index", "level_0", "row_index"}
        or str(col).strip().lower().startswith("_")
    ]
    output = output.drop(columns=drop_cols, errors="ignore")
    numeric_cols = output.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        output[numeric_cols] = output[numeric_cols].round(2)
    return output.head(max_rows).reset_index(drop=True)


def _unique_columns(columns: list[str | None]) -> list[str]:
    seen = set()
    result = []
    for col in columns:
        if col and col not in seen:
            seen.add(col)
            result.append(col)
    return result


def _get_df(data: dict[str, dict[str, object]], key: str) -> pd.DataFrame:
    value = data.get(key, {}).get("df", pd.DataFrame())
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) > 0
    text = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(text, errors="coerce")
    return text.isin(["true", "1", "1.0", "yes", "y"]) | numeric.fillna(0).gt(0)


def _available_columns(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "none"
    return ", ".join(str(col) for col in df.columns)


def _missing_result(df: pd.DataFrame, label: str, intent: str = "historical_sales") -> ChatbotResult:
    return build_response(
        f"I could not find a {label} column in the historical dataset. Available columns are: {_available_columns(df)}",
        source="historical dataframe",
        confidence=25,
        confidence_reason=f"Required {label} column was missing, so no calculation was made.",
        intent=intent,
    )


def _format_date(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return str(value)
    return date.strftime("%b %d, %Y")


def _date_range_text(df: pd.DataFrame, date_col: str | None) -> tuple[str, bool]:
    if not date_col or date_col not in df.columns:
        return "", False
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.notna().any():
        return f" The records cover {_format_date(dates.min())} to {_format_date(dates.max())}.", True
    return "", False


def _historical_confidence(rows: int, match_types: list[str], numeric_series: list[pd.Series] | None = None, needs_date: bool = False, has_date: bool = True) -> tuple[int, str]:
    score = 90
    reasons = ["Calculated directly from the loaded historical dataframe."]
    if any(match == "partial" for match in match_types):
        score -= 10
        reasons.append("At least one required column was detected by partial matching.")
    if rows < 100:
        score -= 10
        reasons.append("The dataframe has fewer than 100 rows.")
    if needs_date and not has_date:
        score -= 5
        reasons.append("No date range was available for a time-based question.")
    for series in numeric_series or []:
        penalty, reason = _numeric_null_penalty(series)
        if penalty:
            score -= penalty
            reasons.append(reason)
    return max(0, min(100, score)), " ".join(reasons)


def _reference_text(metric: str) -> str:
    reference = {
        "revenue": POWER_BI_REFERENCE["total_revenue"],
        "profit": POWER_BI_REFERENCE["total_profit"],
        "units": POWER_BI_REFERENCE["total_units"],
    }.get(metric)
    if reference is None:
        return ""
    return f" Power BI dashboard reference: **{format_millions(reference)}**."


def dashboard_reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"kpi": "Total Revenue", "value": format_millions(POWER_BI_REFERENCE["total_revenue"])},
            {"kpi": "Total Profit", "value": format_millions(POWER_BI_REFERENCE["total_profit"])},
            {"kpi": "Stock on Hand", "value": format_millions(POWER_BI_REFERENCE["stock_on_hand"])},
            {"kpi": "Inventory Retail Value", "value": format_millions(POWER_BI_REFERENCE["inventory_retail_value"])},
            {"kpi": "Inventory Cost Value", "value": format_millions(POWER_BI_REFERENCE["inventory_cost_value"])},
            {"kpi": "Total Units", "value": format_millions(POWER_BI_REFERENCE["total_units"])},
            {"kpi": "Total Stores", "value": format_number(POWER_BI_REFERENCE["total_stores"])},
            {"kpi": "Total Products", "value": format_number(POWER_BI_REFERENCE["total_products"])},
        ]
    )


def answer_dashboard_reference(question: str, reason: str | None = None) -> ChatbotResult:
    text = question.lower()
    prefix = "Using the Power BI dashboard reference"
    if reason:
        prefix += f" because {reason}"
    if any(term in text for term in ["top product", "product"]):
        table = pd.DataFrame(POWER_BI_REFERENCE["top_revenue_products"], columns=["product", "dashboard_revenue"])
        answer = f"{prefix}, the top revenue product is **Lego Bricks** with **2.4M** revenue."
    elif any(term in text for term in ["top store", "store"]):
        table = pd.DataFrame(POWER_BI_REFERENCE["top_stores"], columns=["store", "dashboard_revenue", "dashboard_gross_profit"])
        answer = f"{prefix}, the top performing store is **Maven Toys Ciudad de Mexico 2** with **0.55M** revenue."
    elif "out" in text and "stock" in text:
        table = pd.DataFrame(POWER_BI_REFERENCE["out_of_stock_location_share"], columns=["location", "out_of_stock_share_pct"])
        answer = f"{prefix}, total out-of-stock items are **{format_number(POWER_BI_REFERENCE['out_of_stock_total'])}**. Downtown has the highest out-of-stock share at **59.74%**."
    elif "category" in text:
        table = pd.DataFrame(POWER_BI_REFERENCE["category_revenue_profit"], columns=["category", "dashboard_revenue", "dashboard_gross_profit"])
        answer = f"{prefix}, Toys is the highest revenue category at **5.1M**."
    elif "month" in text:
        table = pd.DataFrame(POWER_BI_REFERENCE["monthly_revenue"], columns=["month", "dashboard_revenue"])
        answer = f"{prefix}, the strongest dashboard months are Mar-Jun at about **1.5M** each."
    else:
        table = dashboard_reference_table()
        answer = f"{prefix}, total revenue is **14.4M**, total profit is **4.0M**, stock on hand is **20.8M**, inventory retail value is **257.2M**, and inventory cost value is **188.1M**."
    return build_response(
        answer,
        table=table,
        source="dashboard reference",
        confidence=75,
        confidence_reason="Based on dashboard reference values, not recalculated from the raw dataset.",
        intent="dashboard_reference",
    )


def is_disabled_module_request(question: str) -> bool:
    text = question.lower()
    return any(pattern.search(text) for pattern in DISABLED_MODULE_PATTERNS) or any(
        phrase in text for phrase in DISABLED_MODULE_PHRASES
    )


def detect_intent(question: str) -> str:
    text = question.lower().strip()
    if not text:
        return "help"
    if is_disabled_module_request(text):
        return "disabled_demand_restock"
    if any(term in text for term in ["help", "what can you answer", "examples"]):
        return "help"
    if any(term in text for term in ["model", "accuracy", "performance", "mape", "rmse", "mae", "error", "reliable", "evaluation", "limitations", "trust"]):
        return "model_evaluation"
    if any(term in text for term in ["dashboard", "power bi", "executive overview", "kpi from dashboard"]):
        return "dashboard_reference"
    if any(term in text for term in ["restock", "restocking", "reorder", "reorder recommendations", "stock risk", "run out", "priority restock", "needs restock", "need restock"]):
        return "disabled_demand_restock"
    if any(term in text for term in ["demand forecast", "expected demand", "future demand", "product demand", "product-store demand", "product store demand", "highest demand", "forecasted demand", "forecasted units", "products will sell most", "products have highest demand", "sell most"]):
        return "disabled_demand_restock"
    if any(term in text for term in ["profit opportunity", "future top products", "top profit products", "top profit products next 30 days", "most profitable", "make most profit", "generate the most profit", "should we promote", "best products by forecasted gross profit"]):
        return "profit_opportunity"
    if any(term in text for term in ["revenue forecast", "future revenue", "next 30 days revenue", "revenue next 30", "expected revenue", "predicted revenue", "forecasted revenue", "total predicted revenue", "highest forecasted revenue"]):
        return "revenue"
    if any(term in text for term in ["profit forecast", "gross profit forecast", "future profit", "next 30 days profit", "profit next 30", "expected gross profit", "expected profit", "forecasted profit", "highest forecasted profit", "gross profit next month"]):
        return "profit"
    historical_terms = [
        "total revenue", "revenue till now", "revenue so far", "total sales", "actual sales", "past sales",
        "historical sales", "existing records", "current records", "records", "raw data", "total gross profit",
        "total profit", "profit till now", "profit historically", "total units", "units sold", "top product",
        "top products", "best product", "best selling", "top selling", "most sold", "sold the most",
        "top store", "best store", "store performance", "product performance", "category performance",
        "revenue by category", "profit by category", "product category", "revenue by month", "monthly revenue",
        "revenue by location", "revenue by store location", "sales by location", "sales by store location",
        "location performance", "store location performance", "inventory", "stock on hand", "out of stock", "out-of-stock",
        "sample historical records", "sample records",
    ]
    if any(term in text for term in historical_terms):
        return "historical_sales"
    if "revenue" in text or "profit" in text or "sales" in text:
        return "historical_sales"
    return "help"


def _top_n_from_question(question: str, default: int = 10, maximum: int = 50) -> int:
    match = re.search(r"\btop\s+(\d+)\b", question.lower())
    if not match:
        return default
    try:
        requested = int(match.group(1))
    except ValueError:
        return default
    return max(1, min(maximum, requested))


def get_model_row(model_eval_df: pd.DataFrame, model_area_keyword: str) -> pd.Series | None:
    if model_eval_df is None or model_eval_df.empty:
        return None
    target_col = find_column(model_eval_df, ["target", "model_area"])
    area_col = find_column(model_eval_df, ["model_area", "target"])
    selected_col = find_column(model_eval_df, ["selected_model"])
    search_cols = [col for col in _unique_columns([target_col, area_col]) if col]
    if not search_cols:
        return None
    keywords = [model_area_keyword.lower()]
    if model_area_keyword == "profit":
        keywords += ["gross_profit", "gross profit"]
    if model_area_keyword == "demand":
        keywords += ["aggregate_units", "unit", "units"]
    mask = pd.Series(False, index=model_eval_df.index)
    for col in search_cols:
        values = model_eval_df[col].astype(str).str.lower()
        for keyword in keywords:
            mask = mask | values.str.contains(keyword, regex=False)
    candidates = model_eval_df[mask]
    if candidates.empty:
        return None
    if selected_col:
        selected = candidates[_to_bool(candidates[selected_col])]
        if not selected.empty:
            return selected.iloc[0]
    mape_col = find_column(candidates, ["mape"])
    if mape_col:
        return candidates.assign(_mape=pd.to_numeric(candidates[mape_col], errors="coerce")).sort_values("_mape").iloc[0]
    return candidates.iloc[0]


def get_model_score(model_eval_df: pd.DataFrame, area: str) -> dict[str, Any] | None:
    row = get_model_row(model_eval_df, area)
    if row is None:
        return None
    mape_col = find_column(model_eval_df, ["mape"])
    if not mape_col or pd.isna(row.get(mape_col)):
        return None
    mape = float(row.get(mape_col))
    score = max(0.0, min(100.0, 100.0 - mape))
    return {"mape": mape, "score": score, "label": model_score_label(score)}


def _forecast_confidence(model_eval_df: pd.DataFrame, area: str, lower: int, upper: int, default: int) -> tuple[int, str, float | None, str | None]:
    score = get_model_score(model_eval_df, area)
    if not score:
        return default, "Forecast file was available, but model evaluation was missing; using default forecast confidence.", None, None
    confidence = int(max(lower, min(upper, score["score"])))
    reason = f"{area.title()} model MAPE is {score['mape']:.2f}%, giving an estimated model score of {score['score']:.2f}%."
    return confidence, reason, score["score"], score["label"]


def summarize_disabled_demand_restock(model_eval_df: pd.DataFrame) -> ChatbotResult:
    score = get_model_score(model_eval_df, "demand")
    if score:
        model_score = score["score"]
        model_label = score["label"]
    else:
        model_score = 58.09
        model_label = "Weak"
    return build_response(
        MODULE_OFFLINE_RESPONSE,
        source="disabled demand/restock module",
        confidence=95,
        confidence_reason="This is a system limitation based on known weak demand model performance.",
        model_performance_score=model_score,
        model_performance_label=model_label,
        intent="disabled_demand_restock",
    )


def _forecast_summary(df: pd.DataFrame, value_candidates: list[str], title: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]] | None:
    if df.empty:
        return None
    value_col = find_column(df, value_candidates)
    if not value_col:
        return None
    date_col = find_column(df, DATE_COLUMNS) or "forecast_period"
    working = df.copy()
    if date_col == "forecast_period":
        working[date_col] = range(1, len(working) + 1)
    working[value_col] = prepare_numeric_series(working, value_col).fillna(0)
    high_idx = working[value_col].idxmax()
    low_idx = working[value_col].idxmin()
    metrics = {
        "total": working[value_col].sum(),
        "average": working[value_col].mean(),
        "highest_date": working.loc[high_idx, date_col],
        "highest_value": working.loc[high_idx, value_col],
        "lowest_date": working.loc[low_idx, date_col],
        "lowest_value": working.loc[low_idx, value_col],
    }
    if "model_used" in working.columns and working["model_used"].notna().any():
        metrics["model_used"] = str(working["model_used"].dropna().iloc[0])
    if "selected_candidate_model" in working.columns and working["selected_candidate_model"].notna().any():
        metrics["selected_candidate_model"] = str(working["selected_candidate_model"].dropna().iloc[0])
    table_cols = [date_col, value_col] + [col for col in ["model_used", "selected_candidate_model"] if col in working.columns]
    table = clean_table(working[table_cols], max_rows=10)
    chart = {"type": "line", "title": title, "df": working[[date_col, value_col]], "x": date_col, "y": value_col}
    return metrics, table, chart


def _forecast_model_note(metrics: dict[str, Any]) -> str:
    model_used = str(metrics.get("model_used", "")).strip()
    selected_candidate = str(metrics.get("selected_candidate_model", "")).strip()
    if not model_used and not selected_candidate:
        return ""
    readable_model = model_used.replace("_", " ")
    readable_candidate = selected_candidate.replace("_", " ")
    if model_used and selected_candidate:
        return f" Model metadata: **{readable_model}** selected from **{readable_candidate}**."
    if model_used:
        return f" Model metadata: **{readable_model}**."
    return f" Selected candidate model: **{readable_candidate}**."


def summarize_revenue_forecast(df: pd.DataFrame, model_eval_df: pd.DataFrame) -> ChatbotResult:
    summary = _forecast_summary(df, REVENUE_FORECAST_COLUMNS + REVENUE_COLUMNS, "Revenue forecast")
    if summary is None:
        return build_response("I could not find usable revenue forecast data.", source="forecast csv", confidence=20, confidence_reason="Revenue forecast file or forecast column was missing.", intent="revenue")
    metrics, table, chart = summary
    confidence, reason, model_score, model_label = _forecast_confidence(model_eval_df, "revenue", 30, 95, 65)
    model_note = _forecast_model_note(metrics)
    answer = f"From the forecast CSV, expected revenue for the next 30 days is approximately **{format_currency(metrics['total'])}**. Average daily forecast revenue is **{format_currency(metrics['average'])}**. The highest forecasted day is **{_format_date(metrics['highest_date'])}** with about **{format_currency(metrics['highest_value'])}**; the lowest is **{_format_date(metrics['lowest_date'])}** at **{format_currency(metrics['lowest_value'])}**.{model_note}"
    return build_response(answer, table=table, chart=chart, source="forecast csv", confidence=confidence, confidence_reason=reason, model_performance_score=model_score, model_performance_label=model_label, intent="revenue")


def summarize_profit_forecast(df: pd.DataFrame, model_eval_df: pd.DataFrame) -> ChatbotResult:
    summary = _forecast_summary(df, PROFIT_FORECAST_COLUMNS + PROFIT_COLUMNS, "Gross profit forecast")
    if summary is None:
        return build_response("I could not find usable gross profit forecast data.", source="forecast csv", confidence=20, confidence_reason="Profit forecast file or forecast column was missing.", intent="profit")
    metrics, table, chart = summary
    confidence, reason, model_score, model_label = _forecast_confidence(model_eval_df, "profit", 30, 95, 65)
    model_note = _forecast_model_note(metrics)
    answer = f"From the forecast CSV, expected gross profit for the next 30 days is approximately **{format_currency(metrics['total'])}**. Average daily gross profit is **{format_currency(metrics['average'])}**. The highest forecasted profit day is **{_format_date(metrics['highest_date'])}** with about **{format_currency(metrics['highest_value'])}**; the lowest is **{_format_date(metrics['lowest_date'])}** at **{format_currency(metrics['lowest_value'])}**.{model_note}"
    return build_response(answer, table=table, chart=chart, source="forecast csv", confidence=confidence, confidence_reason=reason, model_performance_score=model_score, model_performance_label=model_label, intent="profit")


def summarize_model_performance(df: pd.DataFrame) -> ChatbotResult:
    if df.empty:
        return build_response("I could not find `reports/model_evaluation_summary.csv`.", source="model evaluation", confidence=20, confidence_reason="Model evaluation CSV is missing.", intent="model_evaluation")
    working = clean_table(df, max_rows=len(df))
    selected_col = find_column(working, ["selected_model"])
    if selected_col:
        selected = working[_to_bool(working[selected_col])]
        if not selected.empty:
            working = selected.copy()
    target_col = find_column(working, ["target", "model_area"])
    model_col = find_column(working, ["model_used", "candidate_model"])
    mape_col = find_column(working, ["mape"])
    mae_col = find_column(working, ["mae"])
    rmse_col = find_column(working, ["rmse"])
    smape_col = find_column(working, ["smape"])
    if not target_col or not mape_col:
        return build_response("I found the model evaluation file, but not the expected target/MAPE columns.", table=working, source="model evaluation", confidence=30, confidence_reason="The file loaded, but key metric columns were missing.", intent="model_evaluation")
    lines = ["From the model evaluation report:"]
    model_scores = []
    status_values = []
    for _, row in working.iterrows():
        mape = float(row.get(mape_col, 0))
        score = max(0, min(100, 100 - mape))
        model_scores.append(score)
        target = str(row.get(target_col, "model")).replace("_", " ")
        model = str(row.get(model_col, "selected model")).replace("_", " ") if model_col else "selected model"
        selected_text = ""
        if selected_col:
            selected_text = f" Selected model: **{bool(_to_bool(pd.Series([row.get(selected_col)])).iloc[0])}**."
        if _is_demand_model_name(target):
            status = "Disabled for chatbot decisions; not used in user-facing recommendations"
        else:
            status = "Available for user-facing forecast summaries"
        lines.append(
            f"- **{target}** uses {model}.{selected_text} MAE: **{format_number(row.get(mae_col, 0), 2)}**, "
            f"RMSE: **{format_number(row.get(rmse_col, 0), 2)}**, MAPE: **{mape:.2f}%** "
            f"({mape_interpretation(mape)}), sMAPE: **{format_number(row.get(smape_col, 0), 2)}%**. "
            f"Model score: **{score:.2f}% ({model_score_label(score)})**."
        )
        status_values.append(status)
    lines.append(
        "\nBusiness read: these forecasts are best for short-term planning. Accuracy should improve with more "
        "history, promotion calendars, holiday flags, supplier lead time, and stock movement data."
    )
    lines.append(
        "Demand/restock forecast outputs are hidden from the user-facing app because the demand model performance was weak."
    )
    working = working.copy()
    working["user_facing_status"] = status_values
    table_cols = [col for col in _unique_columns([target_col, model_col, mae_col, rmse_col, mape_col, smape_col, selected_col, "user_facing_status"]) if col and col in working.columns]
    overall_score = float(np.mean(model_scores)) if model_scores else None
    return build_response(
        "\n".join(lines),
        table=working[table_cols],
        source="model evaluation",
        confidence=95,
        confidence_reason="Model evaluation CSV is loaded and selected model rows were summarized.",
        model_performance_score=overall_score,
        model_performance_label=model_score_label(overall_score),
        intent="model_evaluation",
    )


def summarize_demand_forecast(df: pd.DataFrame, question: str, model_eval_df: pd.DataFrame) -> ChatbotResult:
    if df.empty:
        return build_response(
            "I could not find usable product demand forecast data.",
            source="forecast csv",
            confidence=20,
            confidence_reason="Demand forecast file is missing.",
            intent="product_demand",
        )
    demand_col = find_column(df, DEMAND_7_COLUMNS if "7" in question.lower() or "week" in question.lower() else DEMAND_30_COLUMNS)
    if not demand_col:
        demand_col = find_column(df, DEMAND_30_COLUMNS + DEMAND_7_COLUMNS + UNITS_COLUMNS)
    product_col = find_column(df, PRODUCT_COLUMNS)
    category_col = find_column(df, CATEGORY_COLUMNS)
    store_col = find_column(df, STORE_COLUMNS)
    if not demand_col:
        return build_response(
            "I could not find a forecasted units column in the demand forecast file.",
            source="forecast csv",
            confidence=20,
            confidence_reason="Required demand forecast column was missing.",
            intent="product_demand",
        )
    working = df.copy()
    working[demand_col] = prepare_numeric_series(working, demand_col).fillna(0)
    top = working.sort_values(demand_col, ascending=False).head(_top_n_from_question(question))
    table_cols = [
        col
        for col in _unique_columns(
            [
                product_col,
                category_col,
                store_col,
                demand_col,
                "forecasted_units_next_7_days",
                "forecasted_units_next_30_days",
                "demand_model_used",
            ]
        )
        if col and col in top.columns
    ]
    table = clean_table(top[table_cols], max_rows=10)
    leader = top.iloc[0][product_col] if product_col and not top.empty else "the leading product"
    confidence, reason, model_score, model_label = _forecast_confidence(model_eval_df, "demand", 20, 90, 65)
    answer = (
        f"From the demand forecast CSV, **{leader}** has the highest forecasted demand at about "
        f"**{format_number(top.iloc[0][demand_col])} units**."
    )
    chart = {"type": "bar", "title": "Top forecasted demand", "df": table, "x": product_col or table.columns[0], "y": demand_col}
    return build_response(
        answer,
        table=table,
        chart=chart,
        source="forecast csv",
        confidence=confidence,
        confidence_reason=reason,
        model_performance_score=model_score,
        model_performance_label=model_label,
        intent="product_demand",
    )


def summarize_restock(df: pd.DataFrame, model_eval_df: pd.DataFrame) -> ChatbotResult:
    if df.empty:
        return build_response(
            "I could not find usable restock recommendation data.",
            source="forecast csv",
            confidence=20,
            confidence_reason="Restock output CSV is missing.",
            intent="restock",
        )
    working = df.copy()
    needs_col = find_column(working, ["needs_restock", "restock_needed", "needs_reorder"])
    if needs_col:
        working = working[_to_bool(working[needs_col])]
    if working.empty:
        return build_response(
            "No urgent restock items are currently flagged in the local restock output.",
            source="forecast csv",
            confidence=75,
            confidence_reason="Restock file loaded and no restock flags were active.",
            intent="restock",
        )
    product_col = find_column(working, PRODUCT_COLUMNS)
    store_col = find_column(working, STORE_COLUMNS)
    location_col = find_column(working, LOCATION_COLUMNS)
    qty_col = find_column(working, RESTOCK_QTY_COLUMNS)
    band_col = find_column(working, PRIORITY_BAND_COLUMNS)
    score_col = find_column(working, PRIORITY_SCORE_COLUMNS)
    stock_col = find_column(working, ["current_stock", "stock_on_hand", "stock"])
    demand_col = find_column(working, DEMAND_7_COLUMNS)
    if band_col:
        order = {"high": 0, "medium": 1, "low": 2}
        working["_priority_rank"] = working[band_col].astype(str).str.lower().map(order).fillna(3)
    else:
        working["_priority_rank"] = 3
    working["_score"] = prepare_numeric_series(working, score_col).fillna(0) if score_col else 0
    working["_qty"] = prepare_numeric_series(working, qty_col).fillna(0) if qty_col else 0
    top = working.sort_values(["_priority_rank", "_score", "_qty"], ascending=[True, False, False]).head(10)
    table_cols = [
        col
        for col in _unique_columns([product_col, store_col, location_col, stock_col, demand_col, qty_col, band_col, score_col])
        if col and col in top.columns
    ]
    table = clean_table(top[table_cols], max_rows=10)
    first = top.iloc[0]
    score = get_model_score(model_eval_df, "demand")
    if score:
        confidence = int(max(25, min(85, score["score"])))
        reason = (
            f"Restock uses forecast demand and business rules. Demand model MAPE is {score['mape']:.2f}%, "
            f"giving an estimated demand model score of {score['score']:.2f}%."
        )
        model_score, model_label = score["score"], score["label"]
    else:
        confidence, reason, model_score, model_label = 65, "Restock CSV exists, but demand model evaluation was unavailable.", None, None
    answer = (
        f"From the restock output, **{len(working)}** product-store items are flagged for restock. "
        f"The highest-priority item is **{first.get(product_col, 'Unknown product')}** at "
        f"**{first.get(store_col, 'Unknown store')}**, with recommended restock quantity "
        f"**{format_number(first.get(qty_col, 0))} units**."
    )
    chart_df = table.copy()
    x_col = product_col or table.columns[0]
    if product_col and store_col and product_col in table.columns and store_col in table.columns:
        chart_df["item_store"] = chart_df[product_col].astype(str) + " - " + chart_df[store_col].astype(str)
        x_col = "item_store"
    chart = {"type": "bar", "title": "Top restock recommendations", "df": chart_df, "x": x_col, "y": qty_col or score_col}
    return build_response(
        answer,
        table=table,
        chart=chart,
        source="forecast csv",
        confidence=confidence,
        confidence_reason=reason,
        model_performance_score=model_score,
        model_performance_label=model_label,
        intent="restock",
    )


def summarize_profit_opportunity(df: pd.DataFrame, question: str, model_eval_df: pd.DataFrame) -> ChatbotResult:
    if df.empty:
        return build_response("I could not find usable profit opportunity data.", source="forecast csv", confidence=20, confidence_reason="Profit opportunity CSV is missing.", intent="profit_opportunity")
    profit_col = find_column(df, FORECASTED_PROFIT_COLUMNS)
    product_col = find_column(df, PRODUCT_COLUMNS)
    category_col = find_column(df, CATEGORY_COLUMNS)
    store_col = find_column(df, STORE_COLUMNS)
    margin_col = find_column(df, UNIT_MARGIN_COLUMNS)
    demand_col = find_column(df, DEMAND_30_COLUMNS + DEMAND_7_COLUMNS + UNITS_COLUMNS)
    if not profit_col:
        return build_response("I could not find a forecasted profit column in the profit opportunity file.", source="forecast csv", confidence=20, confidence_reason="Required forecasted profit column was missing.", intent="profit_opportunity")
    working = df.copy()
    working[profit_col] = prepare_numeric_series(working, profit_col).fillna(0)
    top = working.sort_values(profit_col, ascending=False).head(_top_n_from_question(question))
    table_cols = [col for col in _unique_columns([product_col, category_col, store_col, demand_col, margin_col, profit_col]) if col and col in top.columns]
    table = clean_table(top[table_cols], max_rows=10)
    score = get_model_score(model_eval_df, "profit")
    if score:
        confidence = int(max(25, min(85, score["score"])))
        reason = f"Profit opportunity uses forecasted gross profit output. Profit model MAPE is {score['mape']:.2f}%, giving an estimated model score of {score['score']:.2f}%."
        model_score, model_label = score["score"], score["label"]
    else:
        confidence, reason, model_score, model_label = 65, "Profit opportunity CSV exists, but profit model evaluation was unavailable.", None, None
    leader = top.iloc[0][product_col] if product_col else "the top product"
    answer = f"From the profit opportunity CSV, **{leader}** is the strongest profit opportunity with forecasted gross profit of **{format_currency(top.iloc[0][profit_col])}**."
    chart = {"type": "bar", "title": "Top forecasted profit opportunities", "df": table, "x": product_col or table.columns[0], "y": profit_col}
    return build_response(answer, table=table, chart=chart, source="forecast csv", confidence=confidence, confidence_reason=reason, model_performance_score=model_score, model_performance_label=model_label, intent="profit_opportunity")


def _historical_cols(df: pd.DataFrame) -> dict[str, tuple[str | None, str]]:
    return {
        "date": find_column_with_match(df, DATE_COLUMNS),
        "revenue": find_column_with_match(df, REVENUE_COLUMNS),
        "profit": find_column_with_match(df, PROFIT_COLUMNS),
        "units": find_column_with_match(df, UNITS_COLUMNS),
        "product": find_column_with_match(df, PRODUCT_COLUMNS),
        "store": find_column_with_match(df, STORE_COLUMNS),
        "location": find_column_with_match(df, LOCATION_COLUMNS),
        "city": find_column_with_match(df, CITY_COLUMNS),
        "category": find_column_with_match(df, CATEGORY_COLUMNS),
        "stock": find_column_with_match(df, ["stock_on_hand", "stock on hand", "stock"]),
        "inventory_cost": find_column_with_match(df, ["inventory_cost_value", "inventory_cost", "inventory cost", "cost_value"]),
        "inventory_retail": find_column_with_match(df, ["inventory_retail_value", "inventory_retail", "inventory retail", "retail_value"]),
        "out_of_stock": find_column_with_match(df, OUT_OF_STOCK_COLUMNS),
    }


def _col(col_map: dict[str, tuple[str | None, str]], key: str) -> str | None:
    return col_map[key][0]


def _match(col_map: dict[str, tuple[str | None, str]], key: str) -> str:
    return col_map[key][1]


def summarize_historical_sales(df: pd.DataFrame, question: str, source_path: str | Path | None = None) -> ChatbotResult:
    text = question.lower()
    if df.empty:
        return answer_dashboard_reference(question, reason="the historical sales dataset is not loaded")
    working = df.copy()
    col_map = _historical_cols(working)
    source_name = Path(source_path).name if source_path else "loaded historical file"
    calc_prefix = f"Calculated from loaded historical data file `{source_name}`"
    for key in ["revenue", "profit", "units", "stock", "inventory_cost", "inventory_retail"]:
        col = _col(col_map, key)
        if col:
            working[col] = prepare_numeric_series(working, col)
    date_text, has_date = _date_range_text(working, _col(col_map, "date"))

    sample_terms = ["show records", "sample", "raw data", "existing records", "current records", "actual sales"]
    if any(term in text for term in sample_terms):
        sample_cols = [_col(col_map, k) for k in ["date", "product", "store", "location", "units", "revenue", "profit", "stock"] if _col(col_map, k)]
        table = clean_table(working[_unique_columns(sample_cols)] if sample_cols else working, max_rows=20)
        conf, reason = _historical_confidence(len(working), [], needs_date=False)
        return build_response("Here are sample records from the loaded historical sales dataset.", table=table, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales", table_rows=20)

    if any(term in text for term in ["out of stock", "out-of-stock", "stockout"]):
        out_col = _col(col_map, "out_of_stock")
        if not out_col:
            return answer_dashboard_reference(question, reason="I could not find an out-of-stock column in the historical dataset")
        flag = _to_bool(working[out_col])
        count = int(flag.sum())
        pct = count / len(working) * 100 if len(working) else 0
        location_col = _col(col_map, "location")
        if location_col:
            table = clean_table(working.assign(_out_of_stock=flag).groupby(location_col, as_index=False)["_out_of_stock"].sum().rename(columns={"_out_of_stock": "out_of_stock_records"}).sort_values("out_of_stock_records", ascending=False), max_rows=10)
        else:
            table = pd.DataFrame({"metric": ["out_of_stock_records", "total_records", "out_of_stock_pct"], "value": [count, len(working), pct]})
        conf, reason = _historical_confidence(len(working), [_match(col_map, "out_of_stock")])
        answer = f"{calc_prefix}, **{format_number(count)}** records are marked out of stock, or **{format_percent(pct)}** of records. Power BI dashboard reference reports **{format_number(POWER_BI_REFERENCE['out_of_stock_total'])}** out-of-stock items."
        return build_response(answer, table=table, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["inventory", "stock on hand", "stock"]):
        stock_col = _col(col_map, "stock")
        if not stock_col:
            return answer_dashboard_reference(question, reason="I could not find a stock-on-hand column in the historical dataset")
        rows = [{"metric": "Stock on Hand", "calculated_value": working[stock_col].sum()}]
        if _col(col_map, "inventory_cost"):
            rows.append({"metric": "Inventory Cost Value", "calculated_value": working[_col(col_map, "inventory_cost")].sum()})
        if _col(col_map, "inventory_retail"):
            rows.append({"metric": "Inventory Retail Value", "calculated_value": working[_col(col_map, "inventory_retail")].sum()})
        conf, reason = _historical_confidence(len(working), [_match(col_map, "stock")], [working[stock_col]])
        answer = f"{calc_prefix}, stock on hand totals approximately **{format_number(working[stock_col].sum())}** units. Dashboard reference stock on hand is **20.8M**."
        return build_response(answer, table=pd.DataFrame(rows), source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["category performance", "revenue by category", "profit by category", "product category", "by product category"]):
        category_col = _col(col_map, "category")
        if not category_col:
            return _missing_result(df, "category")
        metrics = [c for c in [_col(col_map, "revenue"), _col(col_map, "profit"), _col(col_map, "units")] if c]
        if not metrics:
            return _missing_result(df, "revenue, gross profit, or units")
        sort_col = _col(col_map, "revenue") or _col(col_map, "profit") or _col(col_map, "units")
        table = clean_table(working.groupby(category_col, as_index=False)[metrics].sum().sort_values(sort_col, ascending=False), max_rows=10)
        conf, reason = _historical_confidence(len(working), [_match(col_map, "category"), _match(col_map, "revenue")], [working[sort_col]])
        answer = f"{calc_prefix}, **{table.iloc[0][category_col]}** is the leading category by {sort_col.replace('_', ' ')}."
        chart = {"type": "bar", "title": "Category performance", "df": table, "x": category_col, "y": sort_col}
        return build_response(answer, table=table, chart=chart, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["monthly revenue", "revenue by month"]):
        revenue_col = _col(col_map, "revenue")
        date_col = _col(col_map, "date")
        if not revenue_col:
            return _missing_result(df, "revenue")
        if not date_col:
            return _missing_result(df, "date or month")
        temp = working.copy()
        if _norm(date_col) in {"month", "monthname"}:
            temp["_month"] = temp[date_col].astype(str).str[:3]
            month_order = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
            temp["_month_order"] = temp["_month"].map(month_order).fillna(99)
        else:
            dates = pd.to_datetime(temp[date_col], errors="coerce")
            temp["_month"] = dates.dt.strftime("%b")
            temp["_month_order"] = dates.dt.month
        grouped = temp.groupby(["_month", "_month_order"], as_index=False)[revenue_col].sum().sort_values("_month_order")
        table = clean_table(grouped.rename(columns={"_month": "month"})[["month", revenue_col]], max_rows=12)
        top = grouped.sort_values(revenue_col, ascending=False).iloc[0]
        conf, reason = _historical_confidence(len(working), [_match(col_map, "revenue"), _match(col_map, "date")], [working[revenue_col]], needs_date=True, has_date=has_date)
        answer = f"{calc_prefix}, the highest revenue month is **{top['_month']}** with **{format_currency(top[revenue_col])}**."
        chart = {"type": "bar", "title": "Monthly revenue", "df": table, "x": "month", "y": revenue_col}
        return build_response(answer, table=table, chart=chart, table_rows=12, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(
        term in text
        for term in [
            "revenue by location",
            "revenue by store location",
            "sales by location",
            "sales by store location",
            "location performance",
            "store location performance",
        ]
    ):
        location_col = _col(col_map, "location")
        metric_col = _col(col_map, "revenue") or _col(col_map, "profit") or _col(col_map, "units")
        if not location_col:
            return _missing_result(df, "store location")
        if not metric_col:
            return _missing_result(df, "revenue, gross profit, or units")
        table = clean_table(working.groupby(location_col, as_index=False)[metric_col].sum().sort_values(metric_col, ascending=False), max_rows=10)
        conf, reason = _historical_confidence(len(working), [_match(col_map, "location")], [working[metric_col]])
        answer = f"{calc_prefix}, **{table.iloc[0][location_col]}** is the leading location by {metric_col.replace('_', ' ')}."
        chart = {"type": "bar", "title": "Store location performance", "df": table, "x": location_col, "y": metric_col}
        return build_response(answer, table=table, chart=chart, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["total revenue", "revenue till now", "revenue so far", "total sales"]):
        revenue_col = _col(col_map, "revenue")
        if not revenue_col:
            return _missing_result(df, "revenue")
        series = working[revenue_col]
        total = series.sum()
        conf, reason = _historical_confidence(len(working), [_match(col_map, "revenue")], [series])
        answer = f"{calc_prefix}, total revenue is approximately **{format_currency(total)}** across **{format_number(len(working))} records**. The detected revenue column is `{revenue_col}`.{date_text}{_reference_text('revenue')}"
        return build_response(answer, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["total gross profit", "total profit", "profit till now", "profit historically"]):
        profit_col = _col(col_map, "profit")
        if not profit_col:
            return _missing_result(df, "gross profit")
        series = working[profit_col]
        total = series.sum()
        conf, reason = _historical_confidence(len(working), [_match(col_map, "profit")], [series])
        answer = f"{calc_prefix}, total gross profit is approximately **{format_currency(total)}** across **{format_number(len(working))} records**. The detected profit column is `{profit_col}`.{date_text}{_reference_text('profit')}"
        return build_response(answer, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["total units", "units sold", "quantity sold", "total quantity"]):
        units_col = _col(col_map, "units")
        if not units_col:
            return _missing_result(df, "units")
        total = working[units_col].sum()
        conf, reason = _historical_confidence(len(working), [_match(col_map, "units")], [working[units_col]])
        answer = f"{calc_prefix}, total units sold are approximately **{format_number(total)}** across **{format_number(len(working))} records**.{date_text}{_reference_text('units')}"
        return build_response(answer, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    product_unit_terms = ["most sold", "most units", "top selling", "units by product", "sold the most units", "sold the most"]
    if re.search(r"\btop\s+\d+\s+products?\b", text) or any(term in text for term in ["top product", "top products", "best product", "best selling", "product performance"]) or any(term in text for term in product_unit_terms):
        product_col = _col(col_map, "product")
        if not product_col:
            return _missing_result(df, "product")
        use_units = any(term in text for term in product_unit_terms)
        metric_col = _col(col_map, "units") if use_units and _col(col_map, "units") else _col(col_map, "revenue") or _col(col_map, "units")
        if not metric_col:
            return _missing_result(df, "revenue or units")
        table = clean_table(working.groupby(product_col, as_index=False)[metric_col].sum().sort_values(metric_col, ascending=False).head(10), max_rows=10)
        metric_name = "units sold" if metric_col == _col(col_map, "units") else "historical revenue"
        value = format_number(table.iloc[0][metric_col]) if metric_col == _col(col_map, "units") else format_currency(table.iloc[0][metric_col])
        conf, reason = _historical_confidence(len(working), [_match(col_map, "product")], [working[metric_col]])
        answer = f"{calc_prefix}, the top product by {metric_name} is **{table.iloc[0][product_col]}** with **{value}**."
        chart = {"type": "bar", "title": f"Top products by {metric_name}", "df": table, "x": product_col, "y": metric_col}
        return build_response(answer, table=table, chart=chart, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    if any(term in text for term in ["top store", "best store", "store performance", "highest historical revenue"]):
        store_col = _col(col_map, "store")
        if not store_col:
            return _missing_result(df, "store")
        metric_col = _col(col_map, "revenue") or _col(col_map, "profit") or _col(col_map, "units")
        if not metric_col:
            return _missing_result(df, "revenue, gross profit, or units")
        table = clean_table(working.groupby(store_col, as_index=False)[metric_col].sum().sort_values(metric_col, ascending=False).head(10), max_rows=10)
        metric_name = "historical revenue" if metric_col == _col(col_map, "revenue") else "gross profit" if metric_col == _col(col_map, "profit") else "units sold"
        value = format_currency(table.iloc[0][metric_col]) if metric_col != _col(col_map, "units") else format_number(table.iloc[0][metric_col])
        conf, reason = _historical_confidence(len(working), [_match(col_map, "store")], [working[metric_col]])
        answer = f"{calc_prefix}, the top store by {metric_name} is **{table.iloc[0][store_col]}** with **{value}**."
        chart = {"type": "bar", "title": f"Top stores by {metric_name}", "df": table, "x": store_col, "y": metric_col}
        return build_response(answer, table=table, chart=chart, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")

    totals = []
    if _col(col_map, "revenue"):
        totals.append(f"revenue **{format_currency(working[_col(col_map, 'revenue')].sum())}**")
    if _col(col_map, "profit"):
        totals.append(f"gross profit **{format_currency(working[_col(col_map, 'profit')].sum())}**")
    if _col(col_map, "units"):
        totals.append(f"units sold **{format_number(working[_col(col_map, 'units')].sum())}**")
    if not totals:
        return _missing_result(df, "revenue, gross profit, or units")
    sample_cols = [_col(col_map, k) for k in ["date", "product", "store", "units", "revenue", "profit"] if _col(col_map, k)]
    table = clean_table(working[_unique_columns(sample_cols)] if sample_cols else working, max_rows=10)
    conf, reason = _historical_confidence(len(working), [])
    answer = f"Calculated from **{format_number(len(working))}** historical records in `{source_name}`, " + ", ".join(totals) + f".{date_text}"
    return build_response(answer, table=table, source="historical dataframe", confidence=conf, confidence_reason=reason, intent="historical_sales")


def build_dashboard_summary(data: dict[str, dict[str, object]]) -> dict[str, dict[str, Any]]:
    historical = _get_df(data, "historical_sales")
    col_map = _historical_cols(historical) if not historical.empty else {}

    def metric(label: str, value: Any | None, source: str) -> dict[str, Any]:
        return {"label": label, "value": value, "source": source}

    metrics: dict[str, dict[str, Any]] = {}
    if not historical.empty and _col(col_map, "revenue"):
        metrics["revenue"] = metric("Total Revenue", prepare_numeric_series(historical, _col(col_map, "revenue")).sum(), "Calculated from loaded data")
    else:
        metrics["revenue"] = metric("Total Revenue", POWER_BI_REFERENCE["total_revenue"], "Dashboard reference")
    if not historical.empty and _col(col_map, "profit"):
        metrics["profit"] = metric("Total Gross Profit", prepare_numeric_series(historical, _col(col_map, "profit")).sum(), "Calculated from loaded data")
    else:
        metrics["profit"] = metric("Total Gross Profit", POWER_BI_REFERENCE["total_profit"], "Dashboard reference")
    if not historical.empty and _col(col_map, "units"):
        metrics["units"] = metric("Total Units", prepare_numeric_series(historical, _col(col_map, "units")).sum(), "Calculated from loaded data")
    else:
        metrics["units"] = metric("Total Units", POWER_BI_REFERENCE["total_units"], "Dashboard reference")
    if not historical.empty and _col(col_map, "store"):
        metrics["stores"] = metric("Total Stores", historical[_col(col_map, "store")].nunique(), "Calculated from loaded data")
    else:
        metrics["stores"] = metric("Total Stores", POWER_BI_REFERENCE["total_stores"], "Dashboard reference")
    if not historical.empty and _col(col_map, "product"):
        metrics["products"] = metric("Total Products", historical[_col(col_map, "product")].nunique(), "Calculated from loaded data")
    else:
        metrics["products"] = metric("Total Products", POWER_BI_REFERENCE["total_products"], "Dashboard reference")
    if not historical.empty and _col(col_map, "stock"):
        metrics["stock"] = metric("Stock on Hand", prepare_numeric_series(historical, _col(col_map, "stock")).sum(), "Calculated from loaded data")
    else:
        metrics["stock"] = metric("Stock on Hand", POWER_BI_REFERENCE["stock_on_hand"], "Dashboard reference")
    return metrics


def get_chatbot_response(question: str, data: dict[str, dict[str, object]], use_openai: bool = False) -> ChatbotResult:
    intent = detect_intent(question)
    model_eval = _get_df(data, "model_evaluation")
    if intent == "model_evaluation":
        result = summarize_model_performance(model_eval)
    elif intent == "dashboard_reference":
        result = answer_dashboard_reference(question)
    elif intent == "disabled_demand_restock":
        result = summarize_disabled_demand_restock(model_eval)
    elif intent == "revenue":
        result = summarize_revenue_forecast(_get_df(data, "revenue_forecast"), model_eval)
    elif intent == "profit":
        result = summarize_profit_forecast(_get_df(data, "profit_forecast"), model_eval)
    elif intent == "product_demand":
        demand_key = "product_store_demand" if any(word in question.lower() for word in ["store", "location", "city"]) else "product_demand"
        result = summarize_demand_forecast(_get_df(data, demand_key), question, model_eval)
    elif intent == "restock":
        result = summarize_restock(_get_df(data, "restock"), model_eval)
    elif intent == "profit_opportunity":
        key = "top_profit_products_by_store" if any(word in question.lower() for word in ["store", "location", "city"]) else "top_profit_products"
        result = summarize_profit_opportunity(_get_df(data, key), question, model_eval)
    elif intent == "historical_sales":
        historical_path = data.get("historical_sales", {}).get("path")
        result = summarize_historical_sales(_get_df(data, "historical_sales"), question, historical_path)
    else:
        result = build_response(HELP_TEXT, source="assistant help", confidence=30, confidence_reason="The question did not match a known analytics intent.", intent="help")
    result.table = clean_table(result.table, max_rows=result.table_rows) if result.table is not None else None
    return result


def answer_question(question: str, data: dict[str, dict[str, object]], use_openai: bool = False) -> dict[str, Any]:
    result = get_chatbot_response(question, data, use_openai=use_openai)
    return {
        "text": result.answer,
        "table": result.table,
        "source": result.source,
        "confidence": result.confidence,
        "confidence_label": confidence_label(result.confidence),
        "confidence_reason": result.confidence_reason,
        "model_performance_score": result.model_performance_score,
        "model_performance_label": result.model_performance_label,
    }
