# Evaluation Results

Revenue and gross profit models are evaluated with chronological holdout data. The pipeline compares a 7-day moving average baseline against exponential smoothing or Holt-Winters where enough history exists.

Demand is evaluated with a simple aggregate 7-day backtest because restock decisions depend on short-term unit demand rather than a long-range demand model.

Lower MAE, RMSE, MAPE, and sMAPE values are better. MAPE is left blank when the actual values are zero and a percentage error would be misleading.

## Selected Model Summary

- revenue: holt_winters_weekly_additive (MAE=2153.64, MAPE=11.28)
- gross_profit: holt_winters_weekly_additive (MAE=656.95, MAPE=12.12)
- aggregate_units: moving_average_7_day (MAE=670.59, MAPE=41.91)
