<div align="center">

# 🚀 Dynamic Retail Demand Forecasting

### *Precision AI for Next-Gen Retail Analytics*

[![Project Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/Dynamic-Retail-Demand-Forecasting-ML/Dynamic-Retail-Demand-Forecasting)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green)](https://flask.palletsprojects.com/)
[![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-orange)](https://developer.mozilla.org/en-US/docs/Web/HTML)

<img src="Presentation/image1.png" alt="Project Banner" width="100%" style="border-radius: 10px; box-shadow: 0 0 20px rgba(255, 0, 204, 0.5);">

</div>

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Project Team](#-project-team)
- [Abstract](#-abstract)
- [Project Architecture](#-project-architecture)
- [Directory Structure](#-directory-structure)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Installation & Setup](#-installation--setup)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Special Thanks](#-special-thanks)

---

## 🎯 Problem Statement
> **"Retail demand fluctuates dramatically due to seasonality shifts, holiday surges, and unpredictable economic conditions. Inaccurate predictions lead to overstocking, costly stockouts, inefficient planning, and declining customer satisfaction."**

*Problem Statement formulated by **Pankaj Agarwal**.*

---

## 👥 Project Team

| Role | Team Member | GitHub / Portfolio |
|:--- |:--- |:--- |
| **Problem Statement** | **Pankaj Agarwal** | [@pank4ja](https://github.com/pank4ja) |
| **Model Creation & Training** | **Swastik Bharadwaj** | [@swastik7781](https://github.com/swastik7781) |
| **Data Cleaning, Frontend, Backend, API, Deployment, Slides** | **Tarnala Sribatsa Patro** | [Portfolio](https://sribatsa.vercel.app/) |

---

## 📝 Abstract
We developed an advanced machine learning system that predicts weekly retail sales with unprecedented accuracy while identifying the holidays that significantly influence consumer purchasing patterns. Our dual approach combines supervised regression for precise sales prediction and classification algorithms for holiday impact analysis. The result is an end-to-end pipeline featuring comprehensive data cleaning, sophisticated feature engineering, and 15 state-of-the-art ML algorithms working in concert.

**Key Achievements:**
- Achieved an **R² Score of 0.91** using Random Forest Regression.
- Successfully classified holiday weeks with **86% Accuracy** using Decision Trees.
- Deployed a fully interactive, futuristic web interface for real-time predictions.

---

## 🏗 Project Architecture

```mermaid
graph TD
    A[Raw Data (Kaggle)] --> B(Data Preprocessing & Cleaning)
    B --> C{Feature Engineering}
    C --> D[Regression Models]
    C --> E[Classification Models]
    D --> F[Sales Prediction]
    E --> G[Holiday Impact Analysis]
    F --> H[Flask API]
    G --> H
    H --> I[Frontend Interface]
    I --> J[User Insights]
```

---

## 📂 Directory Structure

```plaintext
Dynamic-Retail-Demand-Forecasting/
├── Frontend/
│   ├── Presentation/           # Frontend Source Code
│   │   ├── index.html          # Main Application Entry
│   │   ├── style.css           # Custom Design System
│   │   ├── script.js           # Logic & API Integration
│   │   ├── image1.png          # Assets
│   │   └── ...
│   ├── saved_models/           # Trained ML Models (.pkl)
│   │   ├── random_forest_regressor.pkl
│   │   ├── scaler_regression.pkl
│   │   └── ...
│   ├── app.py                  # Flask Backend API
│   ├── requirements.txt        # Python Dependencies
│   └── README.md               # Project Documentation
└── ...
```

---

## ✨ Key Features
- **Advanced ML Models**: Utilizes Random Forest, Gradient Boosting, and Decision Trees for high-accuracy predictions.
- **Dual Analysis**: Performs both Sales Regression (predicting revenue) and Holiday Classification (detecting impact).
- **Interactive UI**: A futuristic, neon-themed frontend for real-time predictions.
- **Robust API**: Flask-based backend serving predictions with low latency.
- **Comprehensive Metrics**: Detailed evaluation using RMSE, MAE, R², Accuracy, and F1 Score.

---

## 💻 Tech Stack
- **Frontend**: HTML5, CSS3 (Custom Design System), JavaScript (Vanilla)
- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: Scikit-learn, NumPy, Pandas, Joblib
- **Deployment**: Vercel (Frontend), Render (Backend)

---

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing
*Led by Tarnala Sribatsa Patro*
- **Handling Missing Values**: Imputed missing markdown data using median values to preserve distribution.
- **Outlier Detection**: Removed extreme anomalies in sales data to prevent model skew.
- **Data Merging**: Combined `train.csv`, `features.csv`, and `stores.csv` into a unified dataset.

### 2. Feature Engineering
- **Date Decomposition**: Extracted `Year`, `Month`, `Week` from timestamps.
- **Categorical Encoding**: Converted Store Types (A, B, C) into numerical values.
- **Lag Features**: Created rolling averages to capture trend continuity.

### 3. Model Training
*Led by Swastik Bharadwaj*
- **Regression**: Trained 7 models including Linear Regression, Ridge, Lasso, and Random Forest.
- **Classification**: Trained 8 models to detect holiday weeks, optimizing for F1 Score.
- **Hyperparameter Tuning**: Used Grid Search to find optimal parameters for Random Forest.

---

## ⚙ Installation & Setup

### 🛠 Prerequisites
Before you begin, ensure you have the following installed:
- **Git**: [Download & Install Git](https://git-scm.com/downloads)
- **Python (3.8+)**: [Download & Install Python](https://www.python.org/downloads/)
  - *Note: During installation, make sure to check the box **"Add Python to PATH"**.*

### 1. Clone the Repository
```bash
git clone https://github.com/Dynamic-Retail-Demand-Forecasting-ML/Dynamic-Retail-Demand-Forecasting.git
cd Dynamic-Retail-Demand-Forecasting
```

### 2. Backend Setup
Navigate to the project directory (root or `Frontend` folder where `app.py` is located):
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r Frontend/requirements.txt
```

### 3. Frontend Setup
The frontend is built with vanilla HTML/CSS/JS, so no `npm install` is required.
Simply open `Frontend/Presentation/index.html` in your browser or use a live server.

### 4. Running Locally
**Start the Backend Server:**
```bash
cd Frontend
python app.py
```
The API will start at `http://localhost:5000`.

**Connect Frontend to Local Backend:**
1. Open `Frontend/Presentation/script.js`.
2. Locate the `fetch` call (around line 141).
3. Change the URL from the deployed link to your local server:
   ```javascript
   // Change this:
   // const response = await fetch('https://dynamic-retail-demand-forecasting-2.onrender.com/api/predict', { ...
   
   // To this:
   const response = await fetch('http://localhost:5000/api/predict', { ...
   ```

---

## 🚀 Deployment

### Frontend Deployment (Vercel)
*Executed by Tarnala Sribatsa Patro*
1. Push your code to GitHub.
2. Go to [Vercel](https://vercel.com) and import your repository.
3. Set the **Root Directory** to `Frontend/Presentation`.
4. Click **Deploy**. Vercel will automatically detect the HTML entry point.

### Backend Deployment (Render)
*Executed by Tarnala Sribatsa Patro*
1. Push your code to GitHub.
2. Go to [Render](https://render.com) and create a new **Web Service**.
3. Connect your repository.
4. Set the **Root Directory** to `Frontend`.
5. Set the **Build Command** to `pip install -r requirements.txt`.
6. Set the **Start Command** to `gunicorn app:app`.
7. Click **Deploy**.

### 🔗 Connecting Frontend to Deployed Backend
Once your backend is live on Render, copy the URL (e.g., `https://your-app.onrender.com`).
1. Open `Frontend/Presentation/script.js`.
2. Update the API endpoint:
   ```javascript
   const response = await fetch('https://your-app.onrender.com/api/predict', { ...
   ```
3. Commit and push the changes to GitHub. Vercel will automatically redeploy the frontend.

---

## 🔌 API Reference

### Predict Sales
`POST /api/predict`

Returns the predicted weekly sales and holiday classification confidence.

**Request Body:**
```json
{
  "store": 1,
  "dept": 1,
  "temperature": 42.31,
  "fuelPrice": 2.572,
  "cpi": 211.096,
  "unemployment": 8.106,
  "storeSize": 151315,
  "isHoliday": 0,
  "month": 2,
  "year": 2025,
  "week": 5,
  "storeType": "A"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "weekly_sales": 24500.50,
    "random_forest_sales": 24200.00,
    "gradient_boosting_sales": 24800.00,
    "sales_category": 0,
    "confidence": 0.92
  }
}
```

---

## 📊 Model Performance

### Regression (Sales Prediction)
| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **5,466.00** | **3,220.58** | **0.9078** |
| Decision Tree | 6,157.99 | 4,013.82 | 0.8829 |
| Gradient Boosting | 6,949.27 | 4,319.22 | 0.8509 |
| Linear Regression | 17,603.12 | 12,160.54 | 0.0434 |

### Classification (Holiday Detection)
| Model | Accuracy | F1 Score | Precision |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | **86.17%** | **72.34%** | **72.34%** |
| Random Forest | 87.94% | 71.67% | 86.87% |
| Gradient Boosting | 75.00% | 62.00% | 62.00% |

---

## 📂 Dataset
The dataset used for this project is the **Walmart Recruiting - Store Sales Forecasting** dataset from Kaggle.
- **Source**: [Kaggle Competition Link](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- **Files**: `train.csv`, `features.csv`, `stores.csv`

---

## �️ Presentation
If you want to view the full project presentation, you can access it here:
👉 **[View Presentation Slides](https://bigdaddyproject.vercel.app/)**

---

## �🙏 Special Thanks
We would like to extend our gratitude to:
- **Dr. Bimal Kumar Meher** (Silicon University) for his guidance and supervision.
- **Kaggle** for providing the dataset.
- **Open Source Community** for the amazing tools and libraries.

---

<div align="center">

### *Made with ❤️ by Group-6 (CSE-B)*

</div>
