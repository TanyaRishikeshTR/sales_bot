"""Reusable forecasting and evaluation utilities."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from src.config import (
    EVALUATION_HOLDOUT_DAYS,
    MIN_SEASONAL_POINTS,
    MOVING_AVERAGE_WINDOW,
    SEASONAL_PERIODS,
)


MOVING_AVERAGE_CANDIDATE = "moving_average_baseline"
EXPONENTIAL_SMOOTHING_CANDIDATE = "exponential_smoothing_holt_winters"


@dataclass
class ForecastResult:
    forecast: pd.Series
    model_used: str


def as_daily_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    fill_value: float = 0.0,
) -> pd.Series:
    """Create a continuous daily time series from a dataframe."""
    if df.empty:
        return pd.Series(dtype=float)

    data = df[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce").fillna(fill_value)
    data = data.dropna(subset=[date_col])

    if data.empty:
        return pd.Series(dtype=float)

    series = data.groupby(date_col)[target_col].sum().sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index).fillna(fill_value).astype(float)
    series.index.name = "date"
    return series


def future_daily_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    if series.empty:
        start = pd.Timestamp.today().normalize()
    else:
        start = pd.to_datetime(series.index.max()) + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq="D")


def clip_nonnegative(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    return series.clip(lower=0).fillna(0)


def moving_average_forecast(
    series: pd.Series,
    horizon: int,
    window: int = MOVING_AVERAGE_WINDOW,
) -> ForecastResult:
    clean = pd.Series(series, dtype=float).dropna()
    index = future_daily_index(clean, horizon)
    if clean.empty:
        average_value = 0.0
    else:
        average_value = float(clean.tail(max(1, window)).mean())
        if math.isnan(average_value):
            average_value = 0.0

    forecast = pd.Series([max(0.0, average_value)] * horizon, index=index)
    return ForecastResult(forecast=forecast, model_used=f"moving_average_{window}_day")


def exponential_smoothing_forecast(
    series: pd.Series,
    horizon: int,
    seasonal_periods: int = SEASONAL_PERIODS,
) -> ForecastResult:
    """Forecast using Holt-Winters where possible, with simple fallbacks."""
    clean = pd.Series(series, dtype=float).dropna()
    clean = clean.clip(lower=0)

    if len(clean) < 3:
        return moving_average_forecast(clean, horizon)

    fit_attempts = []
    if len(clean) >= MIN_SEASONAL_POINTS:
        fit_attempts.append(
            (
                "holt_winters_weekly_additive",
                lambda: ExponentialSmoothing(
                    clean,
                    trend="add" if len(clean) >= 10 else None,
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                ),
            )
        )

    if len(clean) >= 10:
        fit_attempts.append(
            (
                "exponential_smoothing_trend",
                lambda: ExponentialSmoothing(
                    clean,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                ),
            )
        )

    fit_attempts.append(("simple_exponential_smoothing", lambda: SimpleExpSmoothing(clean)))

    for model_name, factory in fit_attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = factory().fit(optimized=True)
                forecast_values = fitted.forecast(horizon)
            forecast = clip_nonnegative(forecast_values)
            forecast.index = future_daily_index(clean, horizon)
            return ForecastResult(forecast=forecast, model_used=model_name)
        except Exception:
            continue

    fallback = moving_average_forecast(clean, horizon)
    fallback.model_used = f"{fallback.model_used}_fallback"
    return fallback


def chronological_train_test_split(
    series: pd.Series,
    test_days: int = EVALUATION_HOLDOUT_DAYS,
    test_fraction: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    clean = pd.Series(series, dtype=float).dropna()
    n = len(clean)
    if n < 5:
        return clean, pd.Series(dtype=float)

    if n >= test_days * 2:
        test_size = test_days
    else:
        test_size = max(1, int(math.ceil(n * test_fraction)))

    if n - test_size < 3:
        test_size = max(1, n - 3)

    return clean.iloc[:-test_size], clean.iloc[-test_size:]


def mae(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def mape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    mask = actual_array != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual_array[mask] - predicted_array[mask]) / actual_array[mask])) * 100)


def smape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_array) + np.abs(predicted_array)
    mask = denominator != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2 * np.abs(predicted_array[mask] - actual_array[mask]) / denominator[mask]) * 100)


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = pd.Series(actual, dtype=float).reset_index(drop=True)
    predicted_values = pd.Series(predicted, dtype=float).reset_index(drop=True)
    length = min(len(actual_values), len(predicted_values))
    actual_values = actual_values.iloc[:length]
    predicted_values = predicted_values.iloc[:length]

    return {
        "mae": mae(actual_values, predicted_values),
        "rmse": rmse(actual_values, predicted_values),
        "mape": mape(actual_values, predicted_values),
        "smape": smape(actual_values, predicted_values),
    }


def forecast_with_candidate(series: pd.Series, horizon: int, candidate_model: str) -> ForecastResult:
    if candidate_model == MOVING_AVERAGE_CANDIDATE:
        return moving_average_forecast(series, horizon)
    if candidate_model == EXPONENTIAL_SMOOTHING_CANDIDATE:
        return exponential_smoothing_forecast(series, horizon)
    raise ValueError(f"Unknown candidate model: {candidate_model}")


def compare_forecast_candidates(series: pd.Series) -> tuple[pd.DataFrame, str]:
    train, test = chronological_train_test_split(series)
    rows: list[dict[str, float | str | int]] = []

    if test.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "candidate_model": MOVING_AVERAGE_CANDIDATE,
                        "model_used": "not_evaluated_insufficient_history",
                        "test_periods": 0,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                        "smape": np.nan,
                        "selected_model": True,
                    }
                ]
            ),
            MOVING_AVERAGE_CANDIDATE,
        )

    for candidate in [MOVING_AVERAGE_CANDIDATE, EXPONENTIAL_SMOOTHING_CANDIDATE]:
        result = forecast_with_candidate(train, len(test), candidate)
        metrics = evaluate_predictions(test, result.forecast)
        rows.append(
            {
                "candidate_model": candidate,
                "model_used": result.model_used,
                "test_periods": len(test),
                **metrics,
            }
        )

    evaluation = pd.DataFrame(rows)
    metric = "mape" if evaluation["mape"].notna().any() else "mae"
    selected_idx = evaluation[metric].astype(float).idxmin()
    evaluation["selected_model"] = False
    evaluation.loc[selected_idx, "selected_model"] = True
    return evaluation, str(evaluation.loc[selected_idx, "candidate_model"])
