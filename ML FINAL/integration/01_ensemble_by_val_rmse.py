# integration/01_ensemble_by_val_rmse.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load ml val preds
ml_val = pd.read_csv("swastik/ml_val_preds.csv", parse_dates=["Date"])
# Load prophet val forecasts
prop_val = pd.read_csv("pankaj/prophet_val_forecasts.csv", parse_dates=["ds"])

# Normalize column names
prop_val = prop_val.rename(columns={"ds":"Date"})

# Merge on Date+Store (and Dept if applicable)
if "Dept" in ml_val.columns:
    merged = pd.merge(ml_val, prop_val, on=["Date","Store"], how="left")
else:
    merged = pd.merge(ml_val, prop_val, on=["Date","Store"], how="left")

# Determine true y
if "Weekly_Sales" in merged.columns:
    y = merged["Weekly_Sales"].values
elif "Sales" in merged.columns:
    y = merged["Sales"].values
else:
    # If target not in merged (because ml_val lacks it), attempt to get from sribatsa sample
    samp = pd.read_csv("sribatsa/sample_small.csv", parse_dates=["Date"]) 
    merged = merged.merge(samp[["Date","Store","Weekly_Sales"]], on=["Date","Store"], how="left")
    y = merged["Weekly_Sales"].values

# Compute RMSE per ML model column and Prophet
possible_ml_models = ["Linear","Ridge","RF","GBR","XGB","LGB"]
ml_model_cols = [c for c in possible_ml_models if c in merged.columns]
if ml_model_cols:
    ml_cols = ml_model_cols
else:
    ml_cols = [c for c in merged.columns if c.endswith("_oof")]

# Compute RMSEs:
rmse_dict = {}
for c in ml_cols:
    preds = merged[c].fillna(0).values
    rmse_dict[c] = mean_squared_error(y, preds, squared=False)
# Prophet RMSE (prophet_yhat)
if "prophet_yhat" in merged.columns:
    rmse_dict["Prophet"] = mean_squared_error(y, merged["prophet_yhat"].fillna(0).values, squared=False)
else:
    rmse_dict["Prophet"] = np.nan

print("Validation RMSEs:", rmse_dict)

# Now compute weights inversely proportional to RMSE
if ml_cols:
    merged["ML_mean"] = merged[ml_cols].mean(axis=1)
    rmse_ml = mean_squared_error(y, merged["ML_mean"].values, squared=False)
else:
    rmse_ml = np.nan

rmse_prophet = rmse_dict.get("Prophet", np.nan)

print("RMSE ML_mean:", rmse_ml, "RMSE Prophet:", rmse_prophet)

weights = [0.5, 0.5]
if not np.isnan(rmse_ml) and not np.isnan(rmse_prophet):
    inv = np.array([1.0/rmse_ml, 1.0/rmse_prophet])
    w = inv / inv.sum()
    weights = w.tolist()
print("Ensemble weights [ML_mean, Prophet]:", weights)

# Apply weights to test predictions
ml_test_path = "swastik/ml_test_preds.csv"
prop_test_path = "pankaj/prophet_test_forecasts.csv"
try:
    ml_test = pd.read_csv(ml_test_path, parse_dates=["Date"])
    prop_test = pd.read_csv(prop_test_path, parse_dates=["ds"]).rename(columns={"ds":"Date"})
    test_ml_cols = [c for c in ml_test.columns if c in possible_ml_models]
    if not test_ml_cols:
        test_ml_cols = [c for c in ml_test.columns if c not in ["Date","Store","Dept"]]
    ml_test["ML_mean"] = ml_test[test_ml_cols].mean(axis=1)
    if "Dept" in ml_test.columns:
        merged_test = pd.merge(ml_test, prop_test, on=["Date","Store"], how="left")
    else:
        merged_test = pd.merge(ml_test, prop_test, on=["Date","Store"], how="left")
    merged_test["prophet_yhat"] = merged_test["prophet_yhat"].fillna(merged_test["ML_mean"])
    merged_test["final_ensemble"] = weights[0]*merged_test["ML_mean"] + weights[1]*merged_test["prophet_yhat"]
    merged_test[["Date","Store","final_ensemble"]].to_csv("integration/final_ensemble_predictions.csv", index=False)
    print("Saved integration/final_ensemble_predictions.csv")
except Exception as e:
    print("Test preds not available or merge failed:", e)
    merged["final_ensemble"] = weights[0]*merged["ML_mean"] + weights[1]*merged["prophet_yhat"].fillna(merged["ML_mean"])
    merged[["Date","Store","final_ensemble"]].to_csv("integration/final_ensemble_predictions_val.csv", index=False)
    print("Saved integration/final_ensemble_predictions_val.csv (from validation)")
