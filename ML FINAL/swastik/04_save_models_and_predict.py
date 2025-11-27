"""
04_save_models_and_predict.py

MODEL PERSISTENCE & PREDICTION INTERFACE
=========================================

This script extends the comprehensive analysis by:
1. Training models and SAVING them to disk (using joblib)
2. Saving the scaler and feature names for consistency
3. Providing a prediction interface for new/future data
4. Loading test.csv and making predictions

SYLLABUS COVERAGE:
- Model Deployment (Practical ML Application)
- Prediction on Unseen Data
- Model Persistence (Industry Best Practice)

AUTHOR: Swastik
DATE: 2025-11-26
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from pathlib import Path

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Import Best Models (based on previous analysis)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# Constants
DATA_PATH_TRAIN = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\train.csv"
DATA_PATH_FEAT  = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\features.csv"
DATA_PATH_STORE = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\stores.csv"
DATA_PATH_TEST  = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\test.csv"

# Model Save Directory
MODEL_DIR = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\swastik\saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

N_ROWS = 5000  # Limit for training
RANDOM_STATE = 42

def load_and_preprocess_data(df_train, df_feat, df_store, is_test=False):
    """
    Preprocesses data consistently for both training and testing.
    
    Args:
        df_train: Training/Test dataframe
        df_feat: Features dataframe
        df_store: Stores dataframe
        is_test: Boolean indicating if this is test data (no Weekly_Sales)
    
    Returns:
        Processed DataFrame
    """
    print(f"Processing {'test' if is_test else 'training'} data...")
    
    # Merge datasets
    df = df_train.merge(df_feat, on=["Store", "Date", "IsHoliday"], how="left") \
                 .merge(df_store, on="Store", how="left")
    
    print(f" -> Merged Shape: {df.shape}")
    
    # Date Processing
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Day_of_Week"] = df["Date"].dt.dayofweek  # NEW: 0=Monday, 6=Sunday
    df["Quarter"] = df["Date"].dt.quarter  # NEW: 1-4
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)  # NEW: Saturday/Sunday
    
    # Drop Date column
    df.drop(columns=["Date"], inplace=True)
    
    # Handle Missing Values (median for numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Encode Categorical Variables
    if "Type" in df.columns:
        le = LabelEncoder()
        df["Type"] = le.fit_transform(df["Type"].astype(str))
    
    # IsHoliday to binary
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    
    # Drop MarkDown columns if they exist (to avoid schema mismatches and sparsity issues)
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df.drop(columns=[c for c in markdown_cols if c in df.columns], inplace=True)
    
    # Ensure all numeric
    df = df.select_dtypes(include=[np.number])
    
    # Fill any remaining NaNs
    df.fillna(0, inplace=True)
    
    print(f" -> Final Shape: {df.shape}")
    return df

def train_and_save_models():
    """
    Trains the best models and saves them along with preprocessing objects.
    """
    print("\n" + "="*80)
    print("STEP 1: TRAINING AND SAVING MODELS")
    print("="*80)
    
    # Load Training Data
    print("\nLoading training datasets...")
    df_train = pd.read_csv(DATA_PATH_TRAIN, nrows=N_ROWS)
    df_feat  = pd.read_csv(DATA_PATH_FEAT)
    df_store = pd.read_csv(DATA_PATH_STORE)
    
    # Preprocess
    df = load_and_preprocess_data(df_train, df_feat, df_store, is_test=False)
    
    # Separate features and targets
    X = df.drop(columns=["Weekly_Sales"])
    y_sales = df["Weekly_Sales"]
    y_holiday = df["IsHoliday"]
    
    # Remove IsHoliday from features for regression
    X_reg = X.drop(columns=["IsHoliday"])
    
    # Remove Store and Dept for classification (to prevent overfitting/leakage)
    # ALSO REMOVE 'Week' and 'Year' to prevent calendar memorization
    cols_to_drop = ['Store', 'Dept', 'Week', 'Year']
    X_class = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    # Scale Features - SEPARATE SCALERS for regression and classification
    print("\nScaling features...")
    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)
    X_reg_scaled = pd.DataFrame(X_reg_scaled, columns=X_reg.columns)
    
    # For classification, keep all features
    scaler_class = StandardScaler()
    X_class_scaled = scaler_class.fit_transform(X_class)
    X_class_scaled = pd.DataFrame(X_class_scaled, columns=X_class.columns)
    
    # Save feature names and scalers
    feature_names_reg = list(X_reg.columns)
    feature_names_class = list(X_class.columns)
    
    joblib.dump(scaler_reg, os.path.join(MODEL_DIR, "scaler_regression.pkl"))
    joblib.dump(scaler_class, os.path.join(MODEL_DIR, "scaler_classification.pkl"))
    joblib.dump(feature_names_reg, os.path.join(MODEL_DIR, "feature_names_regression.pkl"))
    joblib.dump(feature_names_class, os.path.join(MODEL_DIR, "feature_names_classification.pkl"))
    print(" -> Scalers and feature names saved!")

    
    # Train-Test Split for Regression
    X_train_reg, X_test_reg, y_train_sales, y_test_sales = train_test_split(
        X_reg_scaled, y_sales, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Train-Test Split for Classification
    X_train_class, X_test_class, y_train_holiday, y_test_holiday = train_test_split(
        X_class_scaled, y_holiday, test_size=0.2, random_state=RANDOM_STATE, stratify=y_holiday
    )
    
    # ==========================================
    # REGRESSION MODELS (Predict Weekly Sales)
    # ==========================================
    print("\n" + "-"*80)
    print("Training REGRESSION Models (Predicting Weekly Sales)")
    print("-"*80)
    
    # Random Forest Regressor (Best performer from analysis)
    print("\n1. Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_reg.fit(X_train_reg, y_train_sales)
    
    y_pred_rf = rf_reg.predict(X_test_reg)
    rmse_rf = np.sqrt(mean_squared_error(y_test_sales, y_pred_rf))
    r2_rf = r2_score(y_test_sales, y_pred_rf)
    
    print(f"   RMSE: {rmse_rf:.2f}")
    print(f"   R² Score: {r2_rf:.4f}")
    
    # Save model
    joblib.dump(rf_reg, os.path.join(MODEL_DIR, "random_forest_regressor.pkl"))
    print("   ✓ Model saved: random_forest_regressor.pkl")
    
    # Gradient Boosting Regressor
    print("\n2. Training Gradient Boosting Regressor...")
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    gb_reg.fit(X_train_reg, y_train_sales)
    
    y_pred_gb = gb_reg.predict(X_test_reg)
    rmse_gb = np.sqrt(mean_squared_error(y_test_sales, y_pred_gb))
    r2_gb = r2_score(y_test_sales, y_pred_gb)
    
    print(f"   RMSE: {rmse_gb:.2f}")
    print(f"   R² Score: {r2_gb:.4f}")
    
    # Save model
    joblib.dump(gb_reg, os.path.join(MODEL_DIR, "gradient_boosting_regressor.pkl"))
    print("   ✓ Model saved: gradient_boosting_regressor.pkl")
    
    # ==========================================
    # CLASSIFICATION MODELS (Predict IsHoliday)
    # ==========================================
    print("\n" + "-"*80)
    print("Training CLASSIFICATION Models (Predicting IsHoliday)")
    print("-"*80)
    
    # Random Forest Classifier
    print("\n1. Training Random Forest Classifier...")
    rf_class = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_class.fit(X_train_class, y_train_holiday)
    
    accuracy_rf = rf_class.score(X_test_class, y_test_holiday)
    print(f"   Accuracy: {accuracy_rf:.4f}")
    
    # Save model
    joblib.dump(rf_class, os.path.join(MODEL_DIR, "random_forest_classifier.pkl"))
    print("   ✓ Model saved: random_forest_classifier.pkl")
    
    # Gradient Boosting Classifier
    print("\n2. Training Gradient Boosting Classifier...")
    gb_class = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gb_class.fit(X_train_class, y_train_holiday)
    
    accuracy_gb = gb_class.score(X_test_class, y_test_holiday)
    print(f"   Accuracy: {accuracy_gb:.4f}")
    
    # Save model
    joblib.dump(gb_class, os.path.join(MODEL_DIR, "gradient_boosting_classifier.pkl"))
    print("   ✓ Model saved: gradient_boosting_classifier.pkl")
    
    print("\n" + "="*80)
    print("ALL MODELS SAVED SUCCESSFULLY!")
    print(f"Location: {MODEL_DIR}")
    print("="*80)

def predict_on_test_data():
    """
    Loads saved models and makes predictions on test.csv (future data).
    """
    print("\n" + "="*80)
    print("STEP 2: MAKING PREDICTIONS ON TEST DATA")
    print("="*80)
    
    # Check if models exist
    if not os.path.exists(os.path.join(MODEL_DIR, "random_forest_regressor.pkl")):
        print("ERROR: Models not found! Please run train_and_save_models() first.")
        return
    
    # Load Test Data
    print("\nLoading test dataset...")
    try:
        df_test = pd.read_csv(DATA_PATH_TEST, nrows=1000)  # Limit for demo
        df_feat = pd.read_csv(DATA_PATH_FEAT)
        df_store = pd.read_csv(DATA_PATH_STORE)
        print(f" -> Test data loaded: {df_test.shape}")
    except Exception as e:
        print(f"ERROR loading test data: {e}")
        return
    
    # Store original columns for output
    original_cols = df_test[["Store", "Dept"]].copy() if "Dept" in df_test.columns else df_test[["Store"]].copy()
    
    # Preprocess test data
    df_test_processed = load_and_preprocess_data(df_test, df_feat, df_store, is_test=True)
    
    # Load scalers and feature names
    print("\nLoading scalers and feature names...")
    scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
    scaler_class = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))
    feature_names_reg = joblib.load(os.path.join(MODEL_DIR, "feature_names_regression.pkl"))
    feature_names_class = joblib.load(os.path.join(MODEL_DIR, "feature_names_classification.pkl"))
    
    # Prepare features for regression (without IsHoliday)
    X_test_reg = df_test_processed[feature_names_reg]
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    # Prepare features for classification (with all features)
    X_test_class = df_test_processed[feature_names_class]
    X_test_class_scaled = scaler_class.transform(X_test_class)
    
    # ==========================================
    # LOAD MODELS AND PREDICT
    # ==========================================
    print("\n" + "-"*80)
    print("Making Predictions...")
    print("-"*80)
    
    # Load Regression Models
    rf_reg = joblib.load(os.path.join(MODEL_DIR, "random_forest_regressor.pkl"))
    gb_reg = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting_regressor.pkl"))
    
    # Predict Weekly Sales
    print("\n1. Predicting Weekly Sales...")
    pred_sales_rf = rf_reg.predict(X_test_reg_scaled)
    pred_sales_gb = gb_reg.predict(X_test_reg_scaled)
    
    print(f"   Random Forest predictions: {len(pred_sales_rf)} samples")
    print(f"   Gradient Boosting predictions: {len(pred_sales_gb)} samples")
    
    # Load Classification Models
    rf_class = joblib.load(os.path.join(MODEL_DIR, "random_forest_classifier.pkl"))
    gb_class = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting_classifier.pkl"))
    
    # Predict IsHoliday
    print("\n2. Predicting IsHoliday...")
    pred_holiday_rf = rf_class.predict(X_test_class_scaled)
    pred_holiday_gb = gb_class.predict(X_test_class_scaled)
    
    print(f"   Random Forest predictions: {len(pred_holiday_rf)} samples")
    print(f"   Gradient Boosting predictions: {len(pred_holiday_gb)} samples")
    
    # ==========================================
    # CREATE OUTPUT DATAFRAME
    # ==========================================
    print("\n" + "-"*80)
    print("Creating Output File...")
    print("-"*80)
    
    output_df = original_cols.copy()
    output_df["Predicted_Sales_RF"] = pred_sales_rf
    output_df["Predicted_Sales_GB"] = pred_sales_gb
    output_df["Predicted_Sales_Avg"] = (pred_sales_rf + pred_sales_gb) / 2  # Ensemble
    output_df["Predicted_IsHoliday_RF"] = pred_holiday_rf
    output_df["Predicted_IsHoliday_GB"] = pred_holiday_gb
    
    # Save predictions
    output_path = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\swastik\test_predictions.csv"
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Predictions saved to: test_predictions.csv")
    print(f"   Total predictions: {len(output_df)} rows")
    print("\nSample Predictions:")
    print(output_df.head(10))
    
    # Summary Statistics
    print("\n" + "-"*80)
    print("Prediction Summary Statistics")
    print("-"*80)
    print(f"Average Predicted Sales (RF): ${pred_sales_rf.mean():,.2f}")
    print(f"Average Predicted Sales (GB): ${pred_sales_gb.mean():,.2f}")
    print(f"Holiday Predictions (RF): {pred_holiday_rf.sum()} out of {len(pred_holiday_rf)} samples")
    print(f"Holiday Predictions (GB): {pred_holiday_gb.sum()} out of {len(pred_holiday_gb)} samples")

def predict_custom_input(store, dept, temperature, fuel_price, cpi, unemployment, 
                         store_type, store_size, year, month, week, is_holiday=0):
    """
    Makes prediction for a single custom input.
    
    Example Usage:
        predict_custom_input(
            store=1, dept=1, temperature=70.0, fuel_price=3.5,
            cpi=200.0, unemployment=7.5, store_type='A', store_size=150000,
            year=2025, month=12, week=50, is_holiday=1
        )
    """
    print("\n" + "="*80)
    print("CUSTOM INPUT PREDICTION")
    print("="*80)
    
    # Check if models exist
    if not os.path.exists(os.path.join(MODEL_DIR, "random_forest_regressor.pkl")):
        print("ERROR: Models not found! Please run train_and_save_models() first.")
        return
    
    # Create input dataframe
    input_data = {
        "Store": store,
        "Dept": dept,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Type": 0 if store_type == 'A' else (1 if store_type == 'B' else 2),
        "Size": store_size,
        "IsHoliday": is_holiday,
        "Year": year,
        "Month": month,
        "Week": week,
        "Day_of_Week": 0,  # Default Monday
        "Quarter": (month - 1) // 3 + 1,
        "Is_Weekend": 0
    }
    
    df_input = pd.DataFrame([input_data])
    
    print("\nInput Data:")
    print(df_input.T)
    
    # Load scaler and models
    scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
    feature_names_reg = joblib.load(os.path.join(MODEL_DIR, "feature_names_regression.pkl"))
    
    rf_reg = joblib.load(os.path.join(MODEL_DIR, "random_forest_regressor.pkl"))
    
    # Prepare features
    X_input = df_input[feature_names_reg]
    X_input_scaled = scaler_reg.transform(X_input)
    
    # Predict
    prediction = rf_reg.predict(X_input_scaled)[0]
    
    print("\n" + "-"*80)
    print(f"PREDICTED WEEKLY SALES: ${prediction:,.2f}")
    print("-"*80)
    
    return prediction

def main():
    """
    Main execution function.
    """
    print("\n" + "="*80)
    print("MODEL PERSISTENCE & PREDICTION SYSTEM")
    print("="*80)
    
    # Step 1: Train and Save Models
    train_and_save_models()
    
    # Step 2: Predict on Test Data
    predict_on_test_data()
    
    # Step 3: Example Custom Prediction
    print("\n" + "="*80)
    print("EXAMPLE: Custom Input Prediction")
    print("="*80)
    print("\nPredicting sales for Store 1, Dept 1, during holiday week...")
    
    predict_custom_input(
        store=1, 
        dept=1, 
        temperature=65.0, 
        fuel_price=3.2,
        cpi=210.0, 
        unemployment=6.5, 
        store_type='A', 
        store_size=150000,
        year=2025, 
        month=12, 
        week=50, 
        is_holiday=1
    )
    
    print("\n" + "="*80)
    print("COMPLETE! All models saved and predictions generated.")
    print("="*80)

if __name__ == "__main__":
    main()
