# 📘 Workflow: Model Persistence & Prediction (Script 04)

**File:** `04_save_models_and_predict.py`

## 🎯 Purpose
This script handles the **deployment** phase. It bridges the gap between analysis and production by saving the best models and providing a system to generate predictions on new, unseen data.

**Key Objective:** Ensure models used for prediction match the rigorous standards of the analysis phase (same preprocessing, same feature selection).

---

## 🔄 Workflow Steps

### 1. Training & Saving Models
- **Consistency:** Applies the **exact same** preprocessing logic as Script 03.
- **Feature Selection:**
    - **Regression:** Uses all features (excluding `IsHoliday` from input).
    - **Classification:** **DROPS** `Weekly_Sales`, `Week`, and `Year` to match the analysis and prevent overfitting.
- **Training:** Retrains the top models (Random Forest, Gradient Boosting) on the full 8,000-row dataset.
- **Persistence:** Saves models, scalers, and feature lists to `saved_models/` using `joblib`.

### 2. Batch Prediction (Test Data)
- **Input:** Loads `test.csv` (future data).
- **Process:** Loads saved scalers and models.
- **Action:** Generates predictions for `Weekly_Sales` and `IsHoliday`.
- **Output:** Saves results to `test_predictions.csv`.

### 3. Interactive Prediction System
- **Function:** `predict_custom_input()` allows single-instance predictions.
- **UI Integration:** The `prediction_interface.html` file uses this logic (conceptually) to display predictions to users.

---

## 🚀 How to Run
```bash
python 04_save_models_and_predict.py
```

## 🌐 Web Interface
Open `prediction_interface.html` in your browser to see the user-facing prediction dashboard.

## 📂 Output Files
1.  **`saved_models/`**: Contains `.pkl` files for models and scalers.
2.  **`test_predictions.csv`**: Batch predictions for the test dataset.
