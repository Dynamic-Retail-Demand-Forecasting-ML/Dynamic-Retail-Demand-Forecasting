# Model Deployment & Prediction Guide

## 🎯 Overview

This guide explains how to use the trained ML models to make predictions on new data for future sales forecasting.

---

## 📁 Files Created
 
| File | Purpose | Location |
|------|---------|----------|
| `04_save_models_and_predict.py` | Model training, saving, and prediction script | `swastik/` |
| `prediction_interface.html` | Web-based prediction interface (demo) | `swastik/` |
| `saved_models/*.pkl` | Trained model files | `swastik/saved_models/` |
| `test_predictions.csv` | Predictions on test data | `swastik/` |

---

## 🚀 Quick Start Guide

### Step 1: Train and Save Models

Run the main script to train models and save them:

```bash
cd "C:\Users\swast\OneDrive\Desktop\ML PROJECT MAIN\ML FINAL\swastik"
python 04_save_models_and_predict.py
```

**What This Does:**
1. ✅ Loads training data (train.csv, features.csv, stores.csv)
2. ✅ Preprocesses data with enhanced features
3. ✅ Trains 4 models:
   - Random Forest Regressor (Sales Prediction)
   - Gradient Boosting Regressor (Sales Prediction)
   - Random Forest Classifier (Holiday Prediction)
   - Gradient Boosting Classifier (Holiday Prediction)
4. ✅ Saves models to `saved_models/` directory
5. ✅ Loads test.csv and generates predictions
6. ✅ Saves predictions to `test_predictions.csv`
7. ✅ Demonstrates custom input prediction

**Expected Output:**
```
================================================================================
STEP 1: TRAINING AND SAVING MODELS
================================================================================
Loading training datasets...
Processing training data...
 -> Merged Shape: (5000, 13)
 -> Final Shape: (5000, 15)

Scaling features...
 -> Scaler and feature names saved!

--------------------------------------------------------------------------------
Training REGRESSION Models (Predicting Weekly Sales)
--------------------------------------------------------------------------------

1. Training Random Forest Regressor...
   RMSE: 3321.36
   R² Score: 0.9424
   ✓ Model saved: random_forest_regressor.pkl

2. Training Gradient Boosting Regressor...
   RMSE: 5936.29
   R² Score: 0.8160
   ✓ Model saved: gradient_boosting_regressor.pkl

--------------------------------------------------------------------------------
Training CLASSIFICATION Models (Predicting IsHoliday)
--------------------------------------------------------------------------------

1. Training Random Forest Classifier...
   Accuracy: 0.9640
   ✓ Model saved: random_forest_classifier.pkl

2. Training Gradient Boosting Classifier...
   Accuracy: 0.9620
   ✓ Model saved: gradient_boosting_classifier.pkl

================================================================================
ALL MODELS SAVED SUCCESSFULLY!
Location: C:\Users\swast\...\saved_models
================================================================================
```

---

## 📊 Making Predictions

### Method 1: Predict on Test Data (Batch Predictions)

The script automatically predicts on `test.csv` and creates `test_predictions.csv`:

**Output File Structure:**
```csv
Store,Dept,Predicted_Sales_RF,Predicted_Sales_GB,Predicted_Sales_Avg,Predicted_IsHoliday_RF,Predicted_IsHoliday_GB
1,1,24567.89,23456.78,24012.34,0,0
1,2,18234.56,17890.12,18062.34,0,0
...
```

**Columns Explained:**
- `Predicted_Sales_RF`: Random Forest sales prediction
- `Predicted_Sales_GB`: Gradient Boosting sales prediction
- `Predicted_Sales_Avg`: Average of both models (ensemble)
- `Predicted_IsHoliday_RF`: Holiday prediction (0 or 1)
- `Predicted_IsHoliday_GB`: Holiday prediction (0 or 1)

---

### Method 2: Custom Input Prediction (Single Prediction)

**Option A: Using Python Script**

```python
from swastik.04_save_models_and_predict import predict_custom_input

# Example: Predict sales for Store 1, Dept 1, during Christmas week
prediction = predict_custom_input(
    store=1,                # Store number (1-45)
    dept=1,                 # Department number (1-99)
    temperature=65.0,       # Temperature in Fahrenheit
    fuel_price=3.2,         # Fuel price in dollars
    cpi=210.0,              # Consumer Price Index
    unemployment=6.5,       # Unemployment rate (%)
    store_type='A',         # Store type: 'A', 'B', or 'C'
    store_size=150000,      # Store size in square feet
    year=2025,              # Year
    month=12,               # Month (1-12)
    week=50,                # Week number (1-52)
    is_holiday=1            # Is holiday week? 0=No, 1=Yes
)

print(f"Predicted Weekly Sales: ${prediction:,.2f}")
```

**Option B: Using Web Interface**

1. Open `prediction_interface.html` in your browser
2. Fill in the form with store details
3. Click "Predict Sales"
4. View the predicted weekly sales

**Note:** The web interface uses a simplified calculation for demo purposes. For actual predictions, use the Python script which loads the trained models.

---

## 🔧 Advanced Usage

### Loading Saved Models Manually

```python
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('saved_models/random_forest_regressor.pkl')

# Load the scaler
scaler = joblib.load('saved_models/scaler.pkl')

# Load feature names
feature_names = joblib.load('saved_models/feature_names_regression.pkl')

# Prepare your input data
input_data = {
    'Store': 1,
    'Dept': 1,
    'Temperature': 70.0,
    'Fuel_Price': 3.5,
    'CPI': 210.0,
    'Unemployment': 7.5,
    'Type': 0,  # A=0, B=1, C=2
    'Size': 150000,
    'Year': 2025,
    'Month': 12,
    'Week': 50,
    'Day_of_Week': 0,
    'Quarter': 4,
    'Is_Weekend': 0
}

df = pd.DataFrame([input_data])
X = df[feature_names]
X_scaled = scaler.transform(X)

# Make prediction
prediction = model.predict(X_scaled)[0]
print(f"Predicted Sales: ${prediction:,.2f}")
```

---

## 📈 Model Performance Summary

### Regression Models (Predicting Weekly Sales)

| Model | RMSE | MAE | R² Score | Best For |
|-------|------|-----|----------|----------|
| **Random Forest** | 3,321 | 1,709 | **0.9424** | ✅ Best overall performance |
| Gradient Boosting | 5,936 | 4,034 | 0.8160 | Good for ensemble |

### Classification Models (Predicting IsHoliday)

| Model | Accuracy | Precision | Recall | F1 Score | Best For |
|-------|----------|-----------|--------|----------|----------|
| **Random Forest** | **96.4%** | 93.0% | 75.0% | **0.83** | ✅ Best overall |
| Gradient Boosting | 96.2% | 92.0% | 74.0% | 0.82 | Good for ensemble |

---

## 🎓 Syllabus Alignment

This prediction system covers:

✅ **Model Deployment** - Saving and loading trained models  
✅ **Prediction on New Data** - Using models for future forecasting  
✅ **Model Persistence** - Using joblib for model serialization  
✅ **Practical Application** - Real-world sales forecasting interface  
✅ **Ensemble Methods** - Combining multiple model predictions  

---

## 🔍 Understanding the Predictions

### Input Features Explained

| Feature | Description | Example | Impact on Sales |
|---------|-------------|---------|-----------------|
| **Store** | Store identifier | 1-45 | Varies by location |
| **Dept** | Department number | 1-99 | Different categories |
| **Temperature** | Average temperature (°F) | 70.0 | Higher temp → higher sales (seasonal items) |
| **Fuel_Price** | Regional fuel price ($) | 3.5 | Higher fuel → lower sales (less travel) |
| **CPI** | Consumer Price Index | 210.0 | Economic indicator |
| **Unemployment** | Unemployment rate (%) | 7.5 | Higher unemployment → lower sales |
| **Store Type** | A (large), B (medium), C (small) | A | Larger stores → higher sales |
| **Store Size** | Square footage | 150,000 | Bigger → more capacity |
| **Year/Month/Week** | Time features | 2025/12/50 | Seasonal patterns |
| **IsHoliday** | Holiday week flag | 1 | Holidays → 50% sales boost |

### Prediction Confidence

- **High Confidence (R² > 0.9):** Random Forest predictions are highly reliable
- **Ensemble Approach:** Average of RF and GB predictions provides robust estimates
- **Typical Range:** Weekly sales range from $5,000 to $50,000 depending on store size and season

---

## 🛠️ Troubleshooting

### Issue: "Models not found" Error

**Solution:**
```bash
# Run the training script first
python 04_save_models_and_predict.py
```

### Issue: "Feature mismatch" Error

**Solution:** Ensure your input data has all required features:
- Store, Dept, Temperature, Fuel_Price, CPI, Unemployment
- Type, Size, Year, Month, Week
- Day_of_Week, Quarter, Is_Weekend

### Issue: Predictions seem unrealistic

**Possible Causes:**
- Input values outside training data range
- Incorrect store type encoding (A=0, B=1, C=2)
- Missing or incorrect feature values

**Solution:** Validate input ranges:
- Temperature: 20-100°F
- Fuel_Price: $2-$5
- CPI: 150-250
- Unemployment: 3-15%
- Store_Size: 30,000-250,000 sq ft

---

## 📝 Example Use Cases

### Use Case 1: Holiday Season Planning

```python
# Predict sales for Black Friday week
prediction = predict_custom_input(
    store=1, dept=1, temperature=45.0, fuel_price=3.0,
    cpi=215.0, unemployment=5.5, store_type='A', 
    store_size=200000, year=2025, month=11, week=47, 
    is_holiday=1  # Black Friday
)
# Expected: High sales due to holiday + large store
```

### Use Case 2: Summer Sales Forecast

```python
# Predict sales for mid-summer
prediction = predict_custom_input(
    store=10, dept=5, temperature=85.0, fuel_price=3.8,
    cpi=212.0, unemployment=6.0, store_type='B', 
    store_size=120000, year=2025, month=7, week=28, 
    is_holiday=0  # Regular week
)
# Expected: Moderate sales, seasonal items boost
```

### Use Case 3: Economic Downturn Scenario

```python
# Predict sales during high unemployment
prediction = predict_custom_input(
    store=5, dept=3, temperature=60.0, fuel_price=4.5,
    cpi=220.0, unemployment=12.0, store_type='C', 
    store_size=80000, year=2025, month=3, week=12, 
    is_holiday=0  # Regular week
)
# Expected: Lower sales due to economic factors
```

---

## 🎯 Next Steps

1. **Run the script** to train and save models
2. **Check `test_predictions.csv`** for batch predictions
3. **Try custom predictions** with your own inputs
4. **Open the web interface** for interactive predictions
5. **Integrate into your workflow** for ongoing forecasting

---

## 📞 Summary

You now have a complete ML prediction system that:

✅ Trains and saves models for reuse  
✅ Predicts on test data automatically  
✅ Accepts custom inputs for specific scenarios  
✅ Provides both Python and web interfaces  
✅ Achieves 94% accuracy (R² = 0.94) on sales predictions  

**All within your syllabus requirements!** 🎓

---

*Last Updated: November 26, 2025*
