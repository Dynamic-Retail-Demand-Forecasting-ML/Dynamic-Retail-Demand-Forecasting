import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. LOAD DATASETS (Optimized)
# -----------------------------
print("Loading datasets (subset)...")
try:
    # Load only necessary columns and limited rows
    N_ROWS = 5000
    
    # Define columns to load to save memory
    train_cols = ["Store", "Date", "IsHoliday"]
    feat_cols = ["Store", "Date", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    store_cols = ["Store", "Type", "Size"]

    df_features = pd.read_csv(r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\features.csv", usecols=feat_cols)
    df_train    = pd.read_csv(r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\train.csv", nrows=N_ROWS, usecols=train_cols)
    df_stores   = pd.read_csv(r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\stores.csv", usecols=store_cols)
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# -----------------------------
# 2. MERGE DATA
# -----------------------------
# Merge train + features + stores
df = df_train.merge(df_features, on=["Store", "Date", "IsHoliday"], how="left") \
             .merge(df_stores, on="Store", how="left")

print("Merged shape:", df.shape)

# -----------------------------
# 3. PREPROCESSING
# -----------------------------
# Date Features
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week

# Target: IsHoliday
df["IsHoliday"] = df["IsHoliday"].astype(int) 

# Drop Date
df = df.drop(columns=["Date"])

# Handle Missing Values
df = df.fillna(0)

# Encode Categorical Features (Type)
if "Type" in df.columns:
    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])

# Define X and y
y = df["IsHoliday"]
X = df.drop(columns=["IsHoliday"])

print("Target distribution:\n", y.value_counts())

# Ensure numeric and float32
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

# Scale Features
scaler = StandardScaler(copy=False) # In-place scaling
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# 5. DEFINE MODELS
# -----------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=1)
}

# -----------------------------
# 6. TRAIN & EVALUATE
# -----------------------------
results = []

print("\nStarting Training & Evaluation...")
print("-" * 60)

for name, model in models.items():
    print(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        
        print(f"  -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"  -> Failed to train {name}: {e}")

print("-" * 60)

# -----------------------------
# 7. DISPLAY RESULTS
# -----------------------------
results_df = pd.DataFrame(results)
print("\nFinal Metrics Comparison:")
print(results_df)

# Confusion Matrix for Best Model
if not results_df.empty:
    best_model_name = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
    print(f"\nBest Model based on F1 Score: {best_model_name}")

    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\nConfusion Matrix for {best_model_name}:")
    print(cm)

    # Plot Confusion Matrix using Matplotlib
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Holiday', 'Holiday'], rotation=45)
    plt.yticks(tick_marks, ['Not Holiday', 'Holiday'])

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
