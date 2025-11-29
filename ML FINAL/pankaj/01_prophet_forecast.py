# pankaj/01_prophet_forecast.py
import pandas as pd
from prophet import Prophet
import numpy as np

# Load either sample or full cleaned Walmart
try:
    df = pd.read_csv("sribatsa/sample_small.csv", parse_dates=["Date"])
    print("Loaded sample_small.csv")
except Exception:
    df = pd.read_csv("sribatsa/clean_walmart.csv", parse_dates=["Date"])
    print("Loaded clean_walmart.csv")

# For Prophet, produce weekly aggregated series per Store (and Dept optionally)
group_cols = ["Store","Dept"] if "Dept" in df.columns else ["Store"]

# Aggregate weekly sums
agg = df.groupby(group_cols + [pd.Grouper(key="Date", freq="W")])["Weekly_Sales" if "Weekly_Sales" in df.columns else "Sales"].sum().reset_index()
agg = agg.rename(columns={agg.columns[-1]:"y"})

# We'll do a subset of stores for speed (expand later). Choose top stores by total sales.
store_totals = agg.groupby(group_cols[0])["y"].sum().sort_values(ascending=False)
top_stores = store_totals.head(5).index.tolist()  # forecast top 5 stores first

val_frames = []
test_frames = []
for s in top_stores:
    ser = agg[agg[group_cols[0]] == s][["Date","y"]].rename(columns={"Date":"ds"})
    ser = ser.sort_values("ds")
    # Build a small train/validation split: last 8 weeks as validation
    if len(ser) < 26:
        continue
    train = ser.iloc[:-8]
    val = ser.iloc[-8:]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train)
    # Forecast for validation period (same as val ds)
    future_val = val[["ds"]].copy()
    fc_val = m.predict(future_val)
    val_out = pd.DataFrame({
        "Store": [s]*len(fc_val),
        "ds": fc_val["ds"],
        "prophet_yhat": np.round(fc_val["yhat"].values,2)
    })
    val_frames.append(val_out)
    # Forecast next 12 weeks (test)
    future_test = m.make_future_dataframe(periods=12, freq="W")
    fc_test = m.predict(future_test)
    # Keep only the last 12 (future) rows as test forecast
    fc_future = fc_test.tail(12)
    test_out = pd.DataFrame({
        "Store": [s]*len(fc_future),
        "ds": fc_future["ds"],
        "prophet_yhat": np.round(fc_future["yhat"].values,2)
    })
    test_frames.append(test_out)

# Concatenate and save
if val_frames:
    val_df = pd.concat(val_frames, ignore_index=True)
    val_df.to_csv("pankaj/prophet_val_forecasts.csv", index=False)
    print("Saved pankaj/prophet_val_forecasts.csv")
if test_frames:
    test_df = pd.concat(test_frames, ignore_index=True)
    test_df.to_csv("pankaj/prophet_test_forecasts.csv", index=False)
    print("Saved pankaj/prophet_test_forecasts.csv")
