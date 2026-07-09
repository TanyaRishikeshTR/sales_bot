"""Train and evaluate the daily revenue forecast."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DAILY_BUSINESS_METRICS_PATH,
    FORECAST_HORIZONS,
    REVENUE_EVALUATION_PATH,
    REVENUE_FORECAST_PATH,
    VISUALS_DIR,
)
from src.forecast_utils import (
    as_daily_series,
    compare_forecast_candidates,
    forecast_with_candidate,
)
from src.save_outputs import ensure_parent_dir, save_csv


def _plot_forecast(series: pd.Series, forecast: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    plt.figure(figsize=(11, 6))
    recent = series.tail(120)
    plt.plot(recent.index, recent.values, label="historical revenue", linewidth=2)
    plt.plot(
        pd.to_datetime(forecast["date"]),
        forecast["forecasted_revenue"],
        label="forecasted revenue",
        linewidth=2,
    )
    plt.title("Revenue Forecast - Next 30 Days")
    plt.xlabel("date")
    plt.ylabel("revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_revenue_model(
    input_path: str | Path = DAILY_BUSINESS_METRICS_PATH,
    forecast_path: str | Path = REVENUE_FORECAST_PATH,
    evaluation_path: str | Path = REVENUE_EVALUATION_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(input_path)
    series = as_daily_series(daily, "date", "revenue", fill_value=0)

    evaluation, selected_candidate = compare_forecast_candidates(series)
    evaluation.insert(0, "target", "revenue")
    save_csv(evaluation, evaluation_path)

    result = forecast_with_candidate(series, FORECAST_HORIZONS["revenue_days"], selected_candidate)
    forecast = pd.DataFrame(
        {
            "date": result.forecast.index.date,
            "forecasted_revenue": result.forecast.values,
            "model_used": result.model_used,
            "selected_candidate_model": selected_candidate,
        }
    )
    save_csv(forecast, forecast_path)
    _plot_forecast(series, forecast, VISUALS_DIR / "revenue" / "revenue_forecast.png")
    return forecast, evaluation


if __name__ == "__main__":
    train_revenue_model()
