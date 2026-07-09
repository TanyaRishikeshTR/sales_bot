"""Create final model evaluation summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EVALUATION_RESULTS_MD_PATH,
    MODEL_EVALUATION_SUMMARY_PATH,
    PRODUCT_STORE_DAILY_DEMAND_PATH,
    PROFIT_EVALUATION_PATH,
    REVENUE_EVALUATION_PATH,
)
from src.forecast_utils import as_daily_series, evaluate_predictions, moving_average_forecast
from src.save_outputs import save_csv, save_markdown


def _load_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _aggregate_demand_backtest(product_store_path: Path = PRODUCT_STORE_DAILY_DEMAND_PATH) -> pd.DataFrame:
    if not product_store_path.exists():
        return pd.DataFrame()

    demand = pd.read_csv(product_store_path)
    if demand.empty:
        return pd.DataFrame()

    series = as_daily_series(demand, "date", "units", fill_value=0)
    if len(series) < 14:
        return pd.DataFrame(
            [
                {
                    "target": "aggregate_units",
                    "candidate_model": "moving_average_7_day_backtest",
                    "model_used": "not_evaluated_insufficient_history",
                    "test_periods": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "smape": np.nan,
                    "selected_model": True,
                }
            ]
        )

    train = series.iloc[:-7]
    test = series.iloc[-7:]
    forecast = moving_average_forecast(train, 7).forecast
    metrics = evaluate_predictions(test, forecast)
    return pd.DataFrame(
        [
            {
                "target": "aggregate_units",
                "candidate_model": "moving_average_7_day_backtest",
                "model_used": "moving_average_7_day",
                "test_periods": len(test),
                **metrics,
                "selected_model": True,
            }
        ]
    )


def _write_markdown(summary: pd.DataFrame) -> None:
    if "selected_model" in summary.columns:
        selected_mask = summary["selected_model"]
        if selected_mask.dtype != bool:
            selected_mask = selected_mask.astype(str).str.lower().isin(["true", "1", "yes"])
        selected = summary[selected_mask].copy()
    else:
        selected = pd.DataFrame()
    lines = [
        "# Evaluation Results",
        "",
        "Revenue and gross profit models are evaluated with chronological holdout data. "
        "The pipeline compares a 7-day moving average baseline against exponential smoothing "
        "or Holt-Winters where enough history exists.",
        "",
        "Demand is evaluated with a simple aggregate 7-day backtest because restock decisions "
        "depend on short-term unit demand rather than a long-range demand model.",
        "",
        "Lower MAE, RMSE, MAPE, and sMAPE values are better. MAPE is left blank when the actual "
        "values are zero and a percentage error would be misleading.",
        "",
        "## Selected Model Summary",
        "",
    ]
    if selected.empty:
        lines.append("No selected model rows were available. Run the pipeline after extracting data.")
    else:
        for _, row in selected.iterrows():
            lines.append(
                f"- {row.get('target', 'unknown')}: {row.get('model_used', 'unknown')} "
                f"(MAE={row.get('mae', np.nan):.2f}, MAPE={row.get('mape', np.nan):.2f})"
            )

    save_markdown("\n".join(lines) + "\n", EVALUATION_RESULTS_MD_PATH)


def evaluate_all_models(
    revenue_evaluation_path: str | Path = REVENUE_EVALUATION_PATH,
    profit_evaluation_path: str | Path = PROFIT_EVALUATION_PATH,
    output_path: str | Path = MODEL_EVALUATION_SUMMARY_PATH,
) -> pd.DataFrame:
    frames = []
    for model_area, path in [
        ("revenue", Path(revenue_evaluation_path)),
        ("gross_profit", Path(profit_evaluation_path)),
    ]:
        df = _load_if_exists(path)
        if not df.empty:
            df.insert(0, "model_area", model_area)
            frames.append(df)

    demand_eval = _aggregate_demand_backtest()
    if not demand_eval.empty:
        demand_eval.insert(0, "model_area", "demand")
        frames.append(demand_eval)

    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    save_csv(summary, output_path)
    _write_markdown(summary)
    return summary


if __name__ == "__main__":
    evaluate_all_models()
