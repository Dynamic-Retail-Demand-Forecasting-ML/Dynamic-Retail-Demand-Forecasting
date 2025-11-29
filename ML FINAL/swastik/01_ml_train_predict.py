# # swastik/01_ml_train_predict.py
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import xgboost as xgb
# import lightgbm as lgb

# # Paths
# sample_path = "sribatsa/sample_small.csv"
# full_path = "sribatsa/clean_walmart.csv"

# # Load data (use sample if available else full)
# try:
#     df = pd.read_csv(sample_path, parse_dates=["Date"])
#     print("Loaded sample_small.csv")
# except Exception:
#     df = pd.read_csv(full_path, parse_dates=["Date"])
#     print("Loaded clean_walmart.csv")

# # Target detection
# if "Weekly_Sales" in df.columns:
#     TARGET = "Weekly_Sales"
# elif "Sales" in df.columns:
#     TARGET = "Sales"
# else:
#     raise ValueError("No target column found (Weekly_Sales or Sales)")

# # Simple feature selection: numeric columns except the target
# num = df.select_dtypes(include=[np.number]).columns.tolist()
# if TARGET in num:
#     num.remove(TARGET)
# FEATURES = num  # simple numeric features (includes lags created by Sribatsa)

# # train/validation split (time-unaware simple split)
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# X_train = train_df[FEATURES].fillna(0)
# y_train = train_df[TARGET].values
# X_val = val_df[FEATURES].fillna(0)
# y_val = val_df[TARGET].values

# # Define models (use reasonable defaults)
# models = {
#     "Linear": LinearRegression(),
#     "Ridge": Ridge(alpha=1.0),
#     "RF": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
#     "GBR": GradientBoostingRegressor(n_estimators=150, random_state=42),
#     "XGB": xgb.XGBRegressor(n_estimators=150, random_state=42, verbosity=0),
#     "LGB": lgb.LGBMRegressor(n_estimators=150, random_state=42)
# }

# # Train each and collect validation preds and metrics
# val_preds = pd.DataFrame(index=val_df.index)
# metrics = []
# for name, model in models.items():
#     print("Training", name)
#     model.fit(X_train, y_train)
#     p = model.predict(X_val)
#     val_preds[name] = p
#     rmse = mean_squared_error(y_val, p, squared=False)
#     mae = mean_absolute_error(y_val, p)
#     metrics.append({"model": name, "rmse": rmse, "mae": mae})
#     print(name, "RMSE:", rmse, "MAE:", mae)

# metrics_df = pd.DataFrame(metrics)
# metrics_df.to_csv("swastik/ml_val_metrics.csv", index=False)
# val_out = val_df[["Date","Store","Dept"]] if "Dept" in val_df.columns else val_df[["Date","Store"]]
# val_out = val_out.reset_index(drop=True)
# val_out = pd.concat([val_out, val_preds.reset_index(drop=True)], axis=1)
# val_out.to_csv("swastik/ml_val_preds.csv", index=False)
# print("Saved swastik/ml_val_preds.csv and swastik/ml_val_metrics.csv")

# # If you want to produce test predictions for future dates:
# last_dates = df["Date"].drop_duplicates().sort_values().tail(4)
# test_df = df[df["Date"].isin(last_dates)]
# if len(test_df) > 0:
#     X_test = test_df[FEATURES].fillna(0)
#     test_preds = pd.DataFrame(index=test_df.index)
#     for name, model in models.items():
#         test_preds[name] = model.predict(X_test)
#     test_out = test_df[["Date","Store","Dept"]] if "Dept" in test_df.columns else test_df[["Date","Store"]]
#     test_out = test_out.reset_index(drop=True)
#     test_out = pd.concat([test_out, test_preds.reset_index(drop=True)], axis=1)
#     test_out.to_csv("swastik/ml_test_preds.csv", index=False)
#     print("Saved swastik/ml_test_preds.csv (last 4 weeks as test proxy)")
# else:
#     print("Not enough distinct dates to form test proxy; skip test preds.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# LOAD DATASETS
# -----------------------------
print("Loading datasets...")

df_features = pd.read_csv(r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\features.csv")
df_train    = pd.read_csv(r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\train.csv")
df_stores   = pd.read_csv(r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\stores.csv")

print("All datasets loaded successfully!")

# -----------------------------
# MERGE DATA
# -----------------------------
df = df_train.merge(df_features, on=["Store", "Date", "IsHoliday"], how="left") \
             .merge(df_stores, on="Store", how="left")

print("Merged shape:", df.shape)
print(df.head())


# -----------------------------
# BASIC CLEANING (NO PREPROCESSING)
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week

df = df.drop(columns=["Date"])

# -----------------------------
# SEPARATE TARGET + FEATURES
# -----------------------------
y = df["Weekly_Sales"]
X = df.drop(columns=["Weekly_Sales"])

# Convert categoricals automatically
X = pd.get_dummies(X)

print("Final feature shape:", X.shape)

# Handle NaN safely
X = X.fillna(0)

# -----------------------------
# TRAIN–TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Train :", X_train.shape)
print("Test  :", X_test.shape)

# -----------------------------
# MODELS TO RUN
# -----------------------------
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        tree_method="hist",
        max_depth=8
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.1
    )
}

results = {}
predictions = {}

# -----------------------------
# TRAIN AND EVALUATE MODELS
# -----------------------------
print("\n============================")
print("TRAINING MODELS")
print("============================")

for name, model in models.items():
    print(f"\nTraining Model: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = [rmse, mae, r2]

    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R²  : {r2}")

# -----------------------------
# FINAL COMPARISON TABLE
# -----------------------------
print("\n==============================================")
print("FINAL MODEL COMPARISON")
print("==============================================")

for name, metrics in results.items():
    print(f"{name}: RMSE={metrics[0]}   MAE={metrics[1]}   R2={metrics[2]}")

# -----------------------------
# PLOT RESULTS FOR ALL MODELS
# -----------------------------
plt.figure(figsize=(13,6))
for name, pred in predictions.items():
    plt.plot(y_test.values[:200], label="Actual", linewidth=3)
    plt.plot(pred[:200], label=name, alpha=0.7)

plt.title("Model Predictions vs Actual (first 200 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid()
plt.show()

print("\nAll models trained and graph displayed successfully!")




