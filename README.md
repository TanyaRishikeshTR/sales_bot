# Mexico Toy Store ML Forecasting

End-to-end Python forecasting pipeline for Mexico Toy Store sales data stored in Snowflake. The project only reads from Snowflake and saves ML outputs as local CSV files.

## What It Produces

- 30-day revenue forecast
- 30-day gross profit forecast
- 7-day and 30-day product-store demand forecast
- 7-day restock recommendations
- 30-day product profit opportunity rankings
- Model evaluation CSV and markdown summaries

## Folder Structure

```text
data/
  raw/
  processed/
  outputs/
    forecasts/
    restock/
    profit_opportunity/
models/
notebooks/
reports/
src/
visuals/
app/
main.py
requirements.txt
```

## Required `.env` Variables

Place these in the project root `.env` file:

```text
SF_USER=your_user
SF_PASSWORD=your_password
SF_ACCOUNT=your_account
SF_WAREHOUSE=ToyS_WH
SF_DATABASE=TOY_SALES_DB
SF_SCHEMA=ANALYTICS_MARTS
SF_ROLE=optional_role
```

`SF_ROLE` is optional. If it is missing, the connection is created without a role argument.

## Windows Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Full Pipeline

```powershell
python main.py
```

The pipeline runs in this order:

1. Extract joined sales data from Snowflake with a SELECT query.
2. Save a raw local CSV to `data/raw/sales_joined_raw.csv`.
3. Clean and process the dataset.
4. Build daily, product, product-store, and inventory feature datasets.
5. Train revenue and profit forecasts.
6. Forecast product-store demand.
7. Generate restock recommendations.
8. Generate profit opportunity rankings.
9. Save model evaluation outputs.

No tables are created, updated, inserted into, deleted from, or written back to Snowflake.

## Running the Streamlit ML Assistant

After the forecasting pipeline has created the local CSV outputs, launch the chatbot-style business assistant:

```powershell
streamlit run app/streamlit_app.py
```

The assistant reads the CSV files in `data/outputs` and `reports/model_evaluation_summary.csv`. It does not retrain models when opened, and it does not write anything back to Snowflake.

The app can answer questions such as:

- What is the expected revenue for the next 30 days?
- Which products need restock this week?
- Which products may generate the most profit?
- How accurate is the model?

More app-specific guidance is in `app/README_APP.md`.

## Run From an Existing Raw CSV

If Snowflake is unavailable but `data/raw/sales_joined_raw.csv` already exists:

```powershell
python main.py --skip-extract
```

## Final Output CSVs

- `data/outputs/forecasts/revenue_forecast_next_30_days.csv`
- `data/outputs/forecasts/profit_forecast_next_30_days.csv`
- `data/outputs/forecasts/product_store_demand_forecast.csv`
- `data/outputs/forecasts/product_demand_forecast.csv`
- `data/outputs/restock/restock_recommendations_next_7_days.csv`
- `data/outputs/profit_opportunity/top_profit_products_next_30_days.csv`
- `data/outputs/profit_opportunity/top_profit_products_by_store_next_30_days.csv`
- `reports/model_evaluation_summary.csv`

## Evaluation Outputs

- `reports/revenue_model_evaluation.csv`
- `reports/profit_model_evaluation.csv`
- `reports/model_evaluation_summary.csv`
- `reports/evaluation_results.md`
- `visuals/revenue/revenue_forecast.png`
- `visuals/profit/profit_forecast.png`

## Troubleshooting

If Snowflake connection fails, confirm:

- `.env` exists in the project root.
- `SF_USER`, `SF_PASSWORD`, and `SF_ACCOUNT` are correct.
- `SF_WAREHOUSE`, `SF_DATABASE`, and `SF_SCHEMA` match the Snowflake objects.
- Your network or VPN allows Snowflake access.
- The optional `SF_ROLE` has read permission for the listed tables.

If Python imports fail, reactivate the virtual environment and run:

```powershell
python -m pip install -r requirements.txt
```

If the pipeline cannot continue after extraction fails, place a valid raw extract at:

```text
data/raw/sales_joined_raw.csv
```
