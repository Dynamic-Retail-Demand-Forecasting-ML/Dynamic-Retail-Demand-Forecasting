# sribatsa/02_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- PATHS (edit if your files are elsewhere) ---
TRAIN_CSV = "data/walmart/train.csv"       # file you uploaded
FEATURES_CSV = "data/walmart/features.csv"
STORES_CSV = "data/walmart/stores.csv"

# 1) Load
train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"])
features = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
stores = pd.read_csv(STORES_CSV)

# 2) Merge train + features + stores
df = train.merge(features, on=["Store","Date"], how="left")
df = df.merge(stores, on="Store", how="left")

# 3) Basic cleaning
# Fill numeric NA with median
for c in df.select_dtypes(include=[np.number]).columns:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# Fill categorical NA with mode
for c in df.select_dtypes(include=[object]).columns:
    if df[c].isna().any():
        mode = df[c].mode()
        df[c] = df[c].fillna(mode.iat[0] if not mode.empty else "NA")

# 4) Time features
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["week"] = df["Date"].dt.isocalendar().week
df["day"] = df["Date"].dt.day
df["quarter"] = df["Date"].dt.quarter
df["is_weekend"] = (df["Date"].dt.weekday >= 5).astype(int)

# 5) Encode small categorical columns (LabelEncode)
to_encode = ["Type"] if "Type" in df.columns else []
# add any other categorical columns with <=200 unique values
for c in df.select_dtypes(include=[object]).columns:
    if df[c].nunique() <= 200 and c not in to_encode:
        to_encode.append(c)

encoders = {}
for c in to_encode:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    encoders[c] = dict(zip(le.classes_, le.transform(le.classes_)))

# 6) Lag & rolling features: create by Store-Dept if Dept exists else by Store
group_cols = ["Store","Dept"] if "Dept" in df.columns else ["Store"]
df = df.sort_values(group_cols + ["Date"])

# create lags (1,7,30) and rolling mean 7,30 for Weekly_Sales
target_col = "Weekly_Sales"
df["lag_1"] = df.groupby(group_cols)[target_col].shift(1)
df["lag_7"] = df.groupby(group_cols)[target_col].shift(7)
df["lag_30"] = df.groupby(group_cols)[target_col].shift(30)
df["roll_mean_7"] = df.groupby(group_cols)[target_col].shift(1).rolling(7, min_periods=1).mean().reset_index(level=group_cols, drop=True)
df["roll_mean_30"] = df.groupby(group_cols)[target_col].shift(1).rolling(30, min_periods=1).mean().reset_index(level=group_cols, drop=True)

# 7) Fill resulting NA lags with median or 0
for c in ["lag_1","lag_7","lag_30","roll_mean_7","roll_mean_30"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# 8) Save cleaned CSV and a sample small CSV for parallel work
df.to_csv("sribatsa/clean_walmart.csv", index=False)
df.head(20000).to_csv("sribatsa/sample_small.csv", index=False)

print("Saved sribatsa/clean_walmart.csv and sribatsa/sample_small.csv")
