"""Business-friendly response formatting for the chatbot."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def format_money(value: Any) -> str:
    number = _safe_float(value)
    return f"${number:,.0f}"


def format_number(value: Any, decimals: int = 0) -> str:
    number = _safe_float(value)
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}"


def format_units(value: Any) -> str:
    return f"{format_number(value)} units"


def _safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _format_date(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return str(value)
    return date.strftime("%b %d, %Y")


def missing_data_response(dataset_name: str) -> str:
    return (
        f"I could not find usable data for {dataset_name}. "
        "Please run the ML pipeline first or check that the expected CSV exists."
    )


def revenue_summary(metrics: dict[str, Any]) -> str:
    return (
        "For the next 30 days, forecasted revenue is "
        f"{format_money(metrics['total'])}. Average daily revenue is about "
        f"{format_money(metrics['average'])}. The highest forecasted day is "
        f"{_format_date(metrics['highest_date'])} at {format_money(metrics['highest_value'])}, "
        f"and the lowest forecasted day is {_format_date(metrics['lowest_date'])} at "
        f"{format_money(metrics['lowest_value'])}."
    )


def profit_summary(metrics: dict[str, Any]) -> str:
    return (
        "For the next 30 days, forecasted gross profit is "
        f"{format_money(metrics['total'])}. Average daily gross profit is about "
        f"{format_money(metrics['average'])}. The strongest forecasted profit day is "
        f"{_format_date(metrics['highest_date'])} at {format_money(metrics['highest_value'])}, "
        f"and the lowest forecasted profit day is {_format_date(metrics['lowest_date'])} at "
        f"{format_money(metrics['lowest_value'])}."
    )


def restock_recommendations(rows: list[dict[str, Any]], total_flagged: int) -> str:
    if not rows:
        return "No urgent restock items are currently flagged in the local restock output."

    lines = [f"{total_flagged:,} product-store items are flagged for restock. Highest-priority items:"]
    for index, row in enumerate(rows, start=1):
        product = row.get("product_name", row.get("product", "Unknown product"))
        store = row.get("store_name", row.get("store", "Unknown store"))
        location = row.get("store_location", row.get("location", ""))
        quantity = row.get("recommended_restock_quantity", row.get("restock_quantity", 0))
        priority = row.get("priority_band", row.get("priority", "priority not supplied"))
        location_text = f" - {location}" if location else ""
        lines.append(
            f"{index}. {product} - {store}{location_text} - recommended restock: "
            f"{format_units(quantity)} ({priority})"
        )
    lines.append(
        "These are prioritized using the restock flag, priority band, priority score, "
        "and recommended restock quantity from the pipeline output."
    )
    return "\n".join(lines)


def top_profit_products(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return missing_data_response("top profit products")

    lines = ["The products expected to generate the most gross profit over the next 30 days are:"]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. {row.get('product_name', 'Unknown product')} "
            f"({row.get('product_category', 'Unknown category')}) - "
            f"{format_units(row.get('forecasted_units_next_30_days', 0))}, "
            f"margin {format_money(row.get('unit_margin', 0))}, "
            f"forecasted profit {format_money(row.get('forecasted_gross_profit_next_30_days', 0))}"
        )
    return "\n".join(lines)


def product_demand(rows: list[dict[str, Any]], horizon_label: str) -> str:
    if not rows:
        return missing_data_response("product demand forecast")

    lines = [f"Products with the highest forecasted demand for {horizon_label}:"]
    for index, row in enumerate(rows, start=1):
        product = row.get("product_name", row.get("product", "Unknown product"))
        category = row.get("category", row.get("product_category", "Unknown category"))
        units = row.get("forecasted_units_next_30_days", row.get("forecasted_units_next_7_days", row.get("units", 0)))
        lines.append(f"{index}. {product} ({category}) - {format_units(units)}")
    return "\n".join(lines)


def model_evaluation(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return (
            "I do not see model evaluation results yet. After the forecasting pipeline runs, "
            "`reports/model_evaluation_summary.csv` will contain the accuracy metrics."
        )

    lines = [
        "Here is the short model evaluation summary. Lower error values are better:"
    ]
    for row in rows:
        target = row.get("target", row.get("model_area", "model"))
        model_used = row.get("model_used", "selected model")
        lines.append(
            f"- {target}: {model_used}, MAE {format_number(row.get('mae', 0), 2)}, "
            f"MAPE {format_number(row.get('mape', 0), 2)}%"
        )
    lines.append(
        "Business read: these forecasts are best used for short-term planning. The dataset has "
        "one year of history, so simple models were chosen to reduce overfitting."
    )
    return "\n".join(lines)


def limitations_explanation() -> str:
    return (
        "The forecast is useful for short-term planning, not long-term certainty. It uses one year "
        "of sales history, so it may miss unusual promotions, holidays, supplier delays, or new demand "
        "patterns. Accuracy should improve as more history, promotion calendars, holiday flags, and "
        "supplier lead-time data become available."
    )


def methodology_explanation() -> str:
    return (
        "The ML approach is intentionally simple and explainable. Revenue and gross profit compare a "
        "moving average baseline against exponential smoothing or Holt-Winters. Demand forecasts are "
        "used with restock business rules to identify product-store items that may need replenishment."
    )


def help_response() -> str:
    return (
        "You can ask questions like:\n"
        "- What is the expected revenue for the next 30 days?\n"
        "- What is the expected gross profit for next month?\n"
        "- Which products need restock this week?\n"
        "- Which products may generate the most profit?\n"
        "- Which products have highest forecasted demand?\n"
        "- How accurate is the model?\n"
        "- What are the model limitations?"
    )
