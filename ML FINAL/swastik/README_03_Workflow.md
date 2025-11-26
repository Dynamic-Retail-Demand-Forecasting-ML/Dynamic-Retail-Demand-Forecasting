# 📘 Workflow: Comprehensive Analysis (Script 03)

**File:** `03_ml_comprehensive_analysis.py`

## 🎯 Purpose
This script performs a complete, end-to-end analysis of the Walmart dataset, implementing **16 different algorithms** to cover the entire Machine Learning syllabus. It focuses on training, evaluation, and comparison.

---

## 🔄 Workflow Steps

### 1. Data Loading & Preprocessing
- **Input:** Loads `train.csv`, `features.csv`, and `stores.csv`.
- **Merging:** Joins datasets on Store/Date/IsHoliday.
- **Cleaning:** Handles missing values (Forward Fill/Fillna 0).
- **Feature Engineering:** Extracts Year, Month, Week from Date.
- **Encoding:** Converts 'Type' to numeric and 'IsHoliday' to binary.

### 2. Regression Analysis (Predicting Weekly Sales)
Trains and evaluates **7 Regression Models**:
1.  **Linear Regression** (Module 1)
2.  **Ridge Regression** (Module 1)
3.  **Lasso Regression** (Module 1)
4.  **Linear SVR** (Module 3)
5.  **Decision Tree Regressor** (Module 3)
6.  **Random Forest Regressor** (Module 3)
7.  **Gradient Boosting Regressor** (Module 5)

**Metrics:** RMSE, MAE, R² Score.

### 3. Classification Analysis (Predicting Holiday)
Trains and evaluates **9 Classification Models**:
1.  **KNN** (Module 3)
2.  **Logistic Regression** (Module 1)
3.  **LDA** (Module 1)
4.  **Naive Bayes** (Module 3)
5.  **Linear SVC** (Module 3)
6.  **Decision Tree Classifier** (Module 3)
7.  **Random Forest Classifier** (Module 3)
8.  **AdaBoost** (Module 5)
9.  **Gradient Boosting** (Module 5)

**Metrics:** Accuracy, Precision, Recall, F1 Score, ROC AUC.

### 4. Visualization
- **Regression:** Actual vs Predicted Sales plot.
- **Classification:** ROC Curves (Comparison) and Confusion Matrix.

---

## 🚀 How to Run
```bash
python 03_ml_comprehensive_analysis.py
```

## 📤 Output
- **Console:** Detailed performance tables for all 16 models.
- **Plots:** Pop-up windows showing performance graphs.
- **No Files Saved:** This script is for *analysis and comparison only*.
