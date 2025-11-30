from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # Changed from pickle to joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load models and scalers
MODEL_DIR = 'saved_models'

print("Loading models... This may take a moment.")

# Load regression models using joblib (models were saved with joblib)
try:
    rf_regressor = joblib.load(os.path.join(MODEL_DIR, 'random_forest_regressor.pkl'))
    print("✓ Random Forest Regressor loaded")
except Exception as e:
    print(f"✗ Error loading Random Forest Regressor: {e}")
    raise

try:
    gb_regressor = joblib.load(os.path.join(MODEL_DIR, 'gradient_boosting_regressor.pkl'))
    print("✓ Gradient Boosting Regressor loaded")
except Exception as e:
    print(f"✗ Error loading Gradient Boosting Regressor: {e}")
    raise

try:
    scaler_regression = joblib.load(os.path.join(MODEL_DIR, 'scaler_regression.pkl'))
    print("✓ Regression Scaler loaded")
except Exception as e:
    print(f"✗ Error loading Regression Scaler: {e}")
    raise

try:
    feature_names_regression = joblib.load(os.path.join(MODEL_DIR, 'feature_names_regression.pkl'))
    print("✓ Regression Feature Names loaded")
except Exception as e:
    print(f"✗ Error loading Regression Feature Names: {e}")
    raise

# Load classification models using joblib
try:
    rf_classifier = joblib.load(os.path.join(MODEL_DIR, 'random_forest_classifier.pkl'))
    print("✓ Random Forest Classifier loaded")
except Exception as e:
    print(f"✗ Error loading Random Forest Classifier: {e}")
    raise

try:
    gb_classifier = joblib.load(os.path.join(MODEL_DIR, 'gradient_boosting_classifier.pkl'))
    print("✓ Gradient Boosting Classifier loaded")
except Exception as e:
    print(f"✗ Error loading Gradient Boosting Classifier: {e}")
    raise

try:
    scaler_classification = joblib.load(os.path.join(MODEL_DIR, 'scaler_classification.pkl'))
    print("✓ Classification Scaler loaded")
except Exception as e:
    print(f"✗ Error loading Classification Scaler: {e}")
    raise

try:
    feature_names_classification = joblib.load(os.path.join(MODEL_DIR, 'feature_names_classification.pkl'))
    print("✓ Classification Feature Names loaded")
except Exception as e:
    print(f"✗ Error loading Classification Feature Names: {e}")
    raise

print("All models loaded successfully!")
print(f"Regression features: {feature_names_regression}")
print(f"Classification features: {feature_names_classification}")


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract base features from request
        store = float(data.get('store'))
        dept = float(data.get('dept'))
        temperature = float(data.get('temperature'))
        fuel_price = float(data.get('fuelPrice'))
        cpi = float(data.get('cpi'))
        unemployment = float(data.get('unemployment'))
        store_size = float(data.get('storeSize'))
        is_holiday = int(data.get('isHoliday'))
        month = int(data.get('month'))
        year = int(data.get('year', 2025))
        week = int(data.get('week', 1))
        
        # Handle store type encoding (A=0, B=1, C=2)
        store_type = data.get('storeType', 'A')
        if store_type == 'A':
            type_encoded = 0
        elif store_type == 'B':
            type_encoded = 1
        else:  # C
            type_encoded = 2
        
        # Calculate derived features (as done in training script)
        day_of_week = 0  # Default to Monday (we don't have exact date)
        quarter = (month - 1) // 3 + 1  # Calculate quarter from month
        is_weekend = 0  # Default to weekday
        
        # Create complete feature map
        all_features = {
            'Store': store,
            'Dept': dept,
            'Temperature': temperature,
            'Fuel_Price': fuel_price,
            'CPI': cpi,
            'Unemployment': unemployment,
            'Type': type_encoded,
            'Size': store_size,
            'IsHoliday': is_holiday,
            'Year': year,
            'Month': month,
            'Week': week,
            'Day_of_Week': day_of_week,
            'Quarter': quarter,
            'Is_Weekend': is_weekend
        }
        
        print(f"\nReceived prediction request:")
        print(f"All features: {all_features}")
        print(f"Expected regression features: {feature_names_regression}")
        print(f"Expected classification features: {feature_names_classification}")
        
        # Build feature array for REGRESSION (excludes IsHoliday based on training script line 137)
        features_reg = []
        for feature_name in feature_names_regression:
            if feature_name in all_features:
                features_reg.append(float(all_features[feature_name]))
            else:
                print(f"Warning: Feature '{feature_name}' not found, using 0.0")
                features_reg.append(0.0)
        
        # Build feature array for CLASSIFICATION (excludes Store, Dept, Week, Year based on training script line 141-142)
        features_class = []
        for feature_name in feature_names_classification:
            if feature_name in all_features:
                features_class.append(float(all_features[feature_name]))
            else:
                print(f"Warning: Feature '{feature_name}' not found, using 0.0")
                features_class.append(0.0)
        
        # Convert to numpy arrays
        features_reg_array = np.array(features_reg).reshape(1, -1)
        features_class_array = np.array(features_class).reshape(1, -1)
        
        print(f"Regression features shape: {features_reg_array.shape}")
        print(f"Classification features shape: {features_class_array.shape}")
        
        # Scale features
        features_reg_scaled = scaler_regression.transform(features_reg_array)
        features_class_scaled = scaler_classification.transform(features_class_array)
        
        # Make predictions using both regression models
        rf_prediction = rf_regressor.predict(features_reg_scaled)[0]
        gb_prediction = gb_regressor.predict(features_reg_scaled)[0]
        
        # Ensemble prediction (average of both models)
        ensemble_prediction = (rf_prediction + gb_prediction) / 2
        
        # Classification prediction (for holiday detection)
        rf_class = rf_classifier.predict(features_class_scaled)[0]
        gb_class = gb_classifier.predict(features_class_scaled)[0]
        
        # Get probability scores
        rf_proba = rf_classifier.predict_proba(features_class_scaled)[0]
        gb_proba = gb_classifier.predict_proba(features_class_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': {
                'weekly_sales': float(ensemble_prediction),
                'random_forest_sales': float(rf_prediction),
                'gradient_boosting_sales': float(gb_prediction),
                'sales_category': int(rf_class),
                'confidence': float(max(rf_proba))
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\nError in prediction:")
        print(error_details)
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 400


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'regression_features': feature_names_regression,
        'classification_features': feature_names_classification
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
