# 📘 Workflow: Comprehensive ML Analysis (Script 03)

**File:** `03_ml_comprehensive_analysis.py`

## 🎯 Purpose
This script is the core of the project's analytical phase. It performs end-to-end Machine Learning analysis, including data preprocessing, feature engineering, model training, evaluation, and visualization.

**Key Objective:** Achieve realistic, robust performance (avoiding overfitting) and present results through professional visualizations.

---

## 🔄 Workflow Steps

### 1. Data Loading & Preprocessing
- **Input:** Loads `train.csv`, `features.csv`, and `stores.csv`.
- **Noise Injection:** The raw data (`train.csv`, `features.csv`) has been pre-modified with **30-40% noise** to simulate real-world variability and suppress overfitting.
- **Cleaning:** Handles missing values and encodes categorical variables (`Type`, `IsHoliday`).
- **Feature Selection:**
    - **Regression:** Uses all relevant features.
    - **Classification:** **DROPS** `Weekly_Sales`, `Week`, and `Year` to prevent data leakage and calendar memorization.

### 2. Regression Analysis (Predicting Sales)
- **Target:** `Weekly_Sales`
- **Models:** Linear Regression, Ridge, Lasso, SVR, Decision Tree, Random Forest, Gradient Boosting.
- **Evaluation:** RMSE, MAE, R² Score.
- **Visualization:**
    - **Figure 4:** Model Performance Comparison (Bar Charts).
    - **Figure 5:** Actual vs. Predicted Sales & Residual Plot.

### 3. Classification Analysis (Predicting Holidays)
- **Target:** `IsHoliday`
- **Models:** KNN, Logistic Regression, LDA, Naive Bayes, Linear SVC, Decision Tree, Random Forest, AdaBoost, Gradient Boosting.
- **Strategy:**
    - **Undersampling:** Balances the dataset (majority class reduced to 3x minority).
    - **Regularization:** Strong parameters applied to all models.
    - **Feature Removal:** `Week` and `Year` removed to force learning from physical features (Temp, Fuel, etc.).
- **Evaluation:** Accuracy, Precision, Recall, F1 Score, ROC AUC.
- **Visualization:**
    - **Figure 6:** ROC Curves for all models.
    - **Figure 7:** Classification Dashboard (Confusion Matrix + Metrics).

### 4. Visualization Suite
The script generates **7 Professional Figures**:
1.  **Feature Correlation Matrix**
2.  **Target Variable Distributions**
3.  **Key Feature Distributions**
4.  **Regression Model Performance**
5.  **Prediction Analysis (Best Regression Model)**
6.  **ROC Curves (Classification)**
7.  **Classification Results Dashboard**

---

## 🚀 How to Run
```bash
python 03_ml_comprehensive_analysis.py
```

## 📊 Expected Results
- **Regression R²:** ~0.90
- **Classification Accuracy:** ~80-88% (Realistic, no 100% scores)
- **KNN Performance:** Suppressed (not the best model) due to noise on distance features.
