# Walmart Store Sales Forecasting (Time Series + Machine Learning)

## Project Overview
This project builds an end-to-end forecasting pipeline to predict **weekly sales** for Walmart stores and departments using historical transaction data and external economic indicators.

Accurate sales forecasting supports better **inventory planning**, **staffing optimization**, and **promotion strategy**, reducing overstock and improving customer demand fulfillment.

This project is based on the official Kaggle competition dataset:
**Walmart Recruiting - Store Sales Forecasting**

---

## Dataset
Source: Kaggle Competition Dataset  
Files used:
- `train.csv` (historical weekly sales by store and department)
- `test.csv` (future weeks to forecast)
- `features.csv` (markdowns, CPI, unemployment, fuel price, temperature)
- `stores.csv` (store size and type)

---

## Key Objectives
- Perform data preprocessing and merge multiple data sources
- Explore sales trends and holiday impacts using EDA
- Engineer time-based features (lags, rolling averages, date components)
- Train and compare multiple forecasting models:
  - SARIMAX (classical time-series model)
  - Prophet (trend + seasonality forecasting)
  - XGBoost (machine learning regression forecasting)
- Evaluate performance using RMSE, MAE, and MAPE
- Conduct error analysis by store type and department
- Generate predictions for unseen future weeks

---

## Methods & Modeling Approach

### 1. Data Preparation
- Merged training data with store metadata and external economic features
- Handled missing values:
  - MarkDown features filled with 0 (no promotions)
  - CPI, Fuel Price, Unemployment forward-filled within each store

### 2. Feature Engineering
- Extracted date features: Year, Month, Week, Quarter
- Created lag features: `lag_1`, `lag_2`, `lag_4`, `lag_8`
- Created rolling mean features: `roll_mean_4`, `roll_mean_8`, `roll_mean_12`

### 3. Modeling
Models trained:
- **SARIMAX** for aggregate-level forecasting
- **Prophet** for seasonal trend modeling
- **XGBoost** for store-department forecasting

To prevent time leakage:
- Used **time-based train/validation split**
- Applied **TimeSeriesSplit cross-validation**
- Performed hyperparameter tuning using **RandomizedSearchCV**

---

## Model Evaluation
Evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

XGBoost consistently performed best due to its ability to learn nonlinear patterns and incorporate lag-based features.

---

## Error Analysis
Error analysis was performed to understand performance differences across:
- **Store Types (A, B, C)**
- **Departments**

This analysis highlighted which store types and departments were harder to predict due to volatility or promotion effects.

---

## Dashboard & Visualization
An interactive Plotly dashboard was created to visualize:
- overall weekly sales trend
- holiday vs non-holiday sales patterns
- store type sales distribution over time

---

## Results Summary
- Sales exhibit strong seasonal spikes during holiday periods.
- Promotions and store size strongly influence sales.
- XGBoost with lag and rolling features produced the most accurate forecasts.

---

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- XGBoost
- Statsmodels (SARIMAX)
- Prophet
- Plotly
- Matplotlib

---

## How to Run
1. Download dataset from Kaggle:
   https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data
2. Place CSV files in the project directory
3. Run the notebook step-by-step

---

## Future Improvements
- Incorporate hierarchical forecasting (store → department level)
- Add LightGBM for model benchmarking
- Deploy the model via Streamlit or FastAPI

---

## Author
**Precious Ebu**  
Data Scientist | Python | Machine Learning | Forecasting  
LinkedIn: (https://www.linkedin.com/in/precious-ebu) 
Portfolio:(https://github.com/PreciousEbu)
