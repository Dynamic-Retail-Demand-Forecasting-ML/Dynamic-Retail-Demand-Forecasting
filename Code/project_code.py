"""
03_ml_comprehensive_analysis.py

COMPREHENSIVE RETAIL DEMAND FORECASTING & CLASSIFICATION ANALYSIS
=================================================================

This script performs a complete analysis of the Walmart dataset as per the project syllabus.
It covers both:
1.  REGRESSION: Predicting 'Weekly_Sales' (Continuous Target).
2.  CLASSIFICATION: Predicting 'IsHoliday' (Binary Target).

SYLLABUS COVERAGE:
------------------
Module-1:
    - Multiple Linear Regression
    - Shrinkage Methods (Ridge, Lasso)
    - Logistic Regression
    - Linear Discriminant Analysis (LDA)
    - Feature Selection (Implicit via model weights/importance)

Module-2:
    - Performance Metrics (Confusion Matrix, Precision, Recall, ROC Curve)
    - Bias-Variance Trade-off (Observed via Train vs Test scores)

Module-3:
    - Naive Bayes Classifier (Gaussian)
    - SVM for Classification (LinearSVC/SVC)
    - SVM for Regression (LinearSVR/SVR)
    - Decision Trees (Regression & Classification)
    - Random Forest (Regression & Classification)

Module-5:
    - Boosting Methods (AdaBoost, Gradient Boosting)

AUTHOR: Swastik
DATE: 2025-11-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import time
import warnings

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# Regression Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor
)

# Classification Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier
)

# Configuration
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Constants
DATA_PATH_TRAIN = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\train.csv"
DATA_PATH_FEAT  = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\features.csv"
DATA_PATH_STORE = r"C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\data\walmart\stores.csv"

# Memory Optimization: Load limited rows if environment is constrained
N_ROWS = 5000 
RANDOM_STATE = 42

def load_and_preprocess_data():
    """
    Loads data from CSV files, merges them, and performs preprocessing.
    
    Steps:
    1. Load Train, Features, Stores CSVs (using usecols for memory efficiency).
    2. Merge datasets on Store, Date, IsHoliday.
    3. Extract Date features (Year, Month, Week).
    4. Handle Missing Values.
    5. Encode Categorical Variables.
    6. Return processed DataFrame.
    """
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*80)
    
    start_time = time.time()
    
    # Define columns to load to save memory
    train_cols = ["Store", "Date", "IsHoliday", "Weekly_Sales", "Dept"]
    feat_cols = ["Store", "Date", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    store_cols = ["Store", "Type", "Size"]

    print(f"Loading datasets (Limit: {N_ROWS} rows)...")
    try:
        df_train = pd.read_csv(DATA_PATH_TRAIN, nrows=N_ROWS, usecols=train_cols)
        df_feat  = pd.read_csv(DATA_PATH_FEAT, usecols=feat_cols)
        df_store = pd.read_csv(DATA_PATH_STORE, usecols=store_cols)
        print(" -> Datasets loaded successfully.")
    except Exception as e:
        print(f" -> Error loading data: {e}")
        return None

    # Merge Data
    print("Merging datasets...")
    df = df_train.merge(df_feat, on=["Store", "Date", "IsHoliday"], how="left") \
                 .merge(df_store, on="Store", how="left")
    
    print(f" -> Merged Shape: {df.shape}")

    # Date Processing
    print("Processing Date features...")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    
    # Drop original Date column
    df.drop(columns=["Date"], inplace=True)

    # Handle Missing Values
    print("Handling missing values...")
    # Forward fill or fill with 0
    df.fillna(0, inplace=True)

    # Encoding Categorical Variables
    print("Encoding categorical features...")
    # 'Type' is categorical (A, B, C)
    if "Type" in df.columns:
        le = LabelEncoder()
        df["Type"] = le.fit_transform(df["Type"]) # present at store.csv
        print(" -> 'Type' column encoded.")

    # 'IsHoliday' to binary (0/1)
    df["IsHoliday"] = df["IsHoliday"].astype(int) # initially true or false
    print(" -> 'IsHoliday' converted to binary.")

    # Ensure all data is numeric for ML models
    # Drop any remaining object columns just in case
    df = df.select_dtypes(include=[np.number])
    
    print(f" -> Final Processed Shape: {df.shape}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    return df

def run_regression_analysis(df):
    """
    Performs Regression Analysis to predict 'Weekly_Sales'.
    
    Algorithms:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Linear SVR
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor
    """
    print("\n" + "="*80)
    print("STEP 2: REGRESSION ANALYSIS (Target: Weekly_Sales)")
    print("="*80)

    target = "Weekly_Sales"
    if target not in df.columns:
        print(f"Error: Target {target} not found.")
        return

    # Prepare Data
    X = df.drop(columns=[target])
    y = df[target]

    # Scale Features (Important for Linear, Ridge, Lasso, SVR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Linear SVR": LinearSVR(random_state=RANDOM_STATE, max_iter=1000),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_STATE)
    }

    results = []
    predictions = {}

    print("\nTraining Regression Models...")
    print("-" * 60)
    print(f"{'Model':<25} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 60)

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
            print(f"{name:<25} | {rmse:<10.2f} | {mae:<10.2f} | {r2:<10.4f}")

        except Exception as e:
            print(f"{name:<25} | FAILED: {str(e)[:20]}...")

    # Plot Actual vs Predicted (for Best Model based on R2)
    if results:
        best_model = sorted(results, key=lambda x: x['R2'], reverse=True)[0]
        best_name = best_model['Model']
        print(f"\nBest Regression Model: {best_name} (R2: {best_model['R2']:.4f})")

        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values[:100], label="Actual", color='black', linewidth=2)
        plt.plot(predictions[best_name][:100], label=f"Predicted ({best_name})", linestyle='--', linewidth=2)
        plt.title(f"Regression: Actual vs Predicted (First 100 Samples) - {best_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Weekly Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def run_classification_analysis(df):
    """
    Performs Classification Analysis to predict 'IsHoliday'.
    
    Algorithms:
    - KNN
    - Logistic Regression
    - LDA
    - Naive Bayes
    - Linear SVC
    - Decision Tree Classifier
    - Random Forest Classifier
    - AdaBoost
    - Gradient Boosting
    """
    print("\n" + "="*80)
    print("STEP 3: CLASSIFICATION ANALYSIS (Target: IsHoliday)")
    print("="*80)

    target = "IsHoliday"
    if target not in df.columns:
        print(f"Error: Target {target} not found.")
        return

    # Prepare Data
    # Note: We keep Weekly_Sales as a feature for classification
    X = df.drop(columns=[target])
    y = df[target]

    print("Target Distribution:")
    print(y.value_counts())

    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # Define Models
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "Naive Bayes": GaussianNB(),
        "Linear SVC": LinearSVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)
    }

    results = []
    best_f1 = -1
    best_model_name = None
    best_y_pred = None

    print("\nTraining Classification Models...")
    print("-" * 90)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 90)

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results.append({
                "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
            })
            
            print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

            # Track Best Model
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_y_pred = y_pred

            # ROC Curve Plotting
            # Check if model supports predict_proba or decision_function
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob) #_ means threshold  these are not required in the roc curve
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            elif hasattr(model, "decision_function"): # example svm 
                y_prob = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        except Exception as e:
            print(f"{name:<25} | FAILED: {str(e)[:20]}...")

    # Finalize ROC Plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Classification Algorithms')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Confusion Matrix for Best Model
    if best_model_name:
        print(f"\nBest Classification Model: {best_model_name} (F1: {best_f1:.4f})")
        
        cm = confusion_matrix(y_test, best_y_pred)
        
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot Confusion Matrix
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

def main():
    """
    Main execution function.
    """
    print("Starting Comprehensive Analysis...")
    
    # 1. Load Data
    df = load_and_preprocess_data()
    
    if df is not None:
        # 2. Run Regression
        run_regression_analysis(df)
        
        # 3. Run Classification
        run_classification_analysis(df)
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
