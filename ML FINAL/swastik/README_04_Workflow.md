# 📘 Workflow: Model Persistence & Prediction (Script 04)

**File:** `04_save_models_and_predict.py`

## 🎯 Purpose
This script focuses on **Model Deployment and Practical Application**. It takes the best performing models from the analysis phase, trains them on the full dataset, saves them for future use, and generates predictions on new/unseen data (`test.csv`).

---

## 🔄 Workflow Steps

### 1. Training & Saving
- **Input:** Loads training data.
- **Training:** Trains the top 2 models for each task:
    - **Regression:** Random Forest & Gradient Boosting.
    - **Classification:** Random Forest & Gradient Boosting.
- **Persistence:** Saves the trained models, scalers, and feature names using `joblib`.
- **Location:** Models are saved to `swastik/saved_models/`.

### 2. Prediction on Test Data
- **Input:** Loads `test.csv` (Future data without target labels).
- **Preprocessing:** Applies the *exact same* preprocessing and scaling as training.
- **Prediction:** Generates predictions for both Sales and Holiday status.
- **Output:** Saves results to `test_predictions.csv`.

### 3. Custom Input Interface
- Provides a function `predict_custom_input()` to predict sales for any specific scenario (e.g., "What will sales be for Store 1 in December 2025?").

---

## 🚀 How to Run
```bash
python 04_save_models_and_predict.py
```

## 📤 Output Files
1.  **`saved_models/`**: Directory containing `.pkl` files (models, scalers).
2.  **`test_predictions.csv`**: File containing predictions for the test dataset.
3.  **Console Output**: Confirmation of saved models and sample predictions.
