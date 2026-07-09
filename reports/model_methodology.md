# Model Methodology

This project uses one year of Mexico Toy Store sales data, so the modeling approach is intentionally simple and explainable. Daily-level forecasting is used instead of monthly forecasting because a one-year monthly dataset would provide only 12 observations.

Revenue and gross profit forecasts compare a 7-day moving average baseline against Exponential Smoothing or Holt-Winters. Weekly seasonality is attempted only when enough daily history exists. If the statistical model cannot fit safely, the code falls back to a simpler method.

Product and product-store demand forecasts use short-term moving average or Exponential Smoothing logic. Sparse product-store histories fall back to product-level averages per active store, and very sparse product histories can fall back to category averages.

Restock recommendations use a reorder-point rule:

`reorder point = forecasted 7-day demand + safety stock`

Safety stock is set to 20% of forecasted 7-day demand. A product-store combination needs restocking when current stock is less than or equal to the reorder point.

Most profitable products are ranked by forecasted 30-day units multiplied by unit margin. This is a derived business prediction, not a separate complex model.

These forecasts are intended for short-term planning and operational prioritization, not long-term strategic certainty.
