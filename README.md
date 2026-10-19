# Walmart Store Sales Forecasting (Python)

### Time Series Forecasting · Retail Analytics · Machine Learning
![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6B23)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?logo=plotly&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Time_Series-SARIMAX-4051B5)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-4267B2)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

Accurate sales forecasting helps retailers anticipate demand, allocate inventory, plan staffing, and prepare for seasonal fluctuations. Walmart's weekly sales are influenced by factors such as holidays, promotional markdowns, store characteristics, and economic conditions.

In this project, I used Python and time-series forecasting techniques to:

- Analyze historical weekly sales patterns across Walmart stores and departments.
- Investigate holiday-related sales fluctuations and differences between store types.
- Combine sales records with store characteristics and economic and promotional features.
- Develop and tune an XGBoost regression model.
- Compare XGBoost forecasts with SARIMAX and Prophet.
- Evaluate model performance and identify potential applications for retail planning.

## 📈 Business Problem

Retailers must anticipate changes in demand to avoid stockouts, excess inventory, inefficient staffing, and poorly timed promotions.

I approached this problem by developing weekly sales forecasts using historical sales data and contextual variables. I also compared machine-learning and statistical forecasting approaches to determine which performed best under the evaluation methods used in the analysis.

## 🎯 Project Goals

- Explore weekly sales trends, holiday effects, and differences across stores.
- Integrate multiple datasets into a modeling dataset using shared columns.
- Prepare the data and engineer features relevant to sales forecasting.
- Develop a baseline XGBoost model and tune its hyperparameters.
- Compare model performance using MAE, RMSE, MAPE, WAPE, and R².
- Evaluate SARIMAX, Prophet, and aggregated XGBoost forecasts.
- Translate the findings into potential inventory, staffing, and supply-chain applications.

## 🔍 Exploratory Data Analysis (EDA)

### 1. Weekly Sales Trends

I visualized weekly sales over time to examine changes in demand and identify periods of pronounced sales activity. The analysis revealed visible sales peaks around holiday periods, highlighting the importance of accounting for holiday effects in forecasting.

### 2. Holiday Effects

I compared sales patterns during holiday and non-holiday weeks to investigate differences in demand. These comparisons helped establish holiday indicators as relevant contextual variables for the forecasting workflow.

### 3. Store Characteristics

I examined sales across store types and sizes to understand differences in sales levels between stores. Store characteristics were subsequently included as features in the machine-learning workflow.

### 4. Promotional and Economic Variables

I explored the available markdown, temperature, fuel-price, CPI, and unemployment variables as additional context for understanding sales patterns and forecasting demand.

The five markdown columns contained substantial missing values. I replaced missing markdown values with zero, treating them as periods with no recorded markdown. This was a modeling assumption rather than confirmation that every missing value represented an absence of promotional activity.

## 🤖 Forecasting Workflow

### 📌 1. Data Integration and Preparation

I worked with four datasets:

| Dataset        | Dimensions  | Purpose                                  |
| -------------- | ----------- | ---------------------------------------- |
| `train.csv`    | 421,570 × 5 | Historical sales records                 |
| `test.csv`     | 115,064 × 4 | Records requiring forecasts              |
| `features.csv` | 8,190 × 12  | Holiday, markdown, and economic features |
| `stores.csv`   | 45 × 3      | Store type and size information          |

I joined the relevant datasets using their shared columns, including `Store`, `Date`, and `IsHoliday` where applicable. This combined sales records with store characteristics and contextual variables for analysis and modeling.

The modeling data covered 5 February 2010 to 26 October 2012.

### 📌 2. Time-Based Validation

For the store-and-department XGBoost workflow, I split the data chronologically at 1 January 2012.

| Dataset    | Rows    |
| ---------- | ------- |
| Training   | 294,132 |
| Validation | 127,438 |

I preserved the temporal order of the observations rather than randomly splitting them, reflecting the forecasting task of using historical data to predict future sales.

### 📌 3. Models Evaluated

I developed and compared the following forecasting approaches:

- Baseline XGBoost: Established an initial machine-learning benchmark.
- Tuned XGBoost: Used `TimeSeriesSplit` and `RandomizedSearchCV` to optimize model hyperparameters.
- Prophet: Provided a statistical forecasting benchmark on aggregated weekly sales.
- SARIMAX: Provided an additional time-series benchmark on the aggregated weekly sales series.

The hyperparameter search evaluated 15 parameter configurations across five time-series folds, resulting in 75 fits.

The selected XGBoost parameters were:

```
{    "subsample": 0.8,    "n_estimators": 800,    "max_depth": 6,    "learning_rate": 0.05,    "colsample_bytree": 0.6}
```

### 📌 4. Evaluation Metrics

I evaluated the forecasts using:

- MAE: Measures average absolute prediction error.
- RMSE: Penalizes larger prediction errors more heavily.
- MAPE: Measures average absolute percentage error.
- WAPE: Measures total absolute error relative to total actual sales.
- R²: Measures performance relative to a mean-based prediction baseline.

I interpreted the metrics together rather than selecting a model based on a single score. MAPE also requires caution when actual sales values are zero or close to zero.

## 🏆 Model Performance Summary

The analysis produced two distinct evaluation results: one for the store-and-department XGBoost workflow and another for forecasts aggregated to total weekly sales. These results represent different evaluation levels and should not be compared directly.

**Fair Model Comparison**: I aggregated the XGBoost store-and-department-level predictions into total weekly sales to align its output with the weekly forecasts generated by Prophet and SARIMAX. This enabled a more consistent comparison of the three models at the same forecasting level using MAE, RMSE, MAPE, WAPE, and R².

### 1. Store-and-Department XGBoost Performance

| Metric | Baseline XGBoost  | Tuned XGBoost |
| ------ | ----------------- | ------------- |
| RMSE   | 2,977.27          | 3,093.43      |
| MAE    | Not reported here | 1,477.33      |
| WAPE   | Not reported here | 9.41%         |
| R²     | 0.982             | 0.9804        |

The hyperparameter search improved the best cross-validation RMSE from approximately 4,885.50 to 4,598.08. However, the tuned model did not outperform the baseline on the reported holdout RMSE or R².

This demonstrated that improved cross-validation performance did not necessarily translate into better performance on the holdout data.

### 2. Aggregated Weekly Forecast Comparison

The store-and-department XGBoost predictions were aggregated into weekly sales forecasts and compared with Prophet and SARIMAX over 43 weekly observations.

| Model                | RMSE       | MAE        | MAPE   | WAPE   | R²      |
| -------------------- | ---------- | ---------- | ------ | ------ | ------- |
| XGBoost (Aggregated) | 1,752,574  | 1,227,584  | 2.61%  | 2.64%  | 0.482   |
| Prophet              | 2,423,342  | 1,800,569  | 3.85%  | 3.87%  | 0.010   |
| SARIMAX              | 10,893,740 | 10,570,583 | 22.72% | 22.73% | -19.004 |

### 🥇 Best Performer: Aggregated XGBoost

XGBoost achieved the lowest RMSE, MAE, MAPE, and WAPE in the aggregated weekly comparison.

- RMSE: Approximately 1.75 million.
- MAE: Approximately 1.23 million.
- WAPE: 2.64%, compared with 3.87% for Prophet and 22.73% for SARIMAX.
- R²: Approximately 0.482.

Prophet ranked second based on the reported error metrics. SARIMAX performed substantially worse, with a negative R² indicating that it performed worse than a mean-based prediction baseline on the evaluated observations.

Important distinction: Although aggregated XGBoost outperformed the two statistical benchmarks, the tuned store-and-department model did not outperform the baseline on the reported holdout RMSE and R². Model selection therefore depends on the evaluation level and intended forecasting use.

## 📊 Key Findings

1. Holiday periods showed pronounced sales peaks. Holiday effects are important to consider when forecasting weekly demand.
2. Store characteristics provided useful context. Sales levels varied across stores, making store type and size relevant forecasting inputs.
3. Promotional and economic information enriched the analysis. Markdown and economic variables provided additional context for sales variation.
4. Aggregated XGBoost outperformed Prophet and SARIMAX. It achieved the lowest error metrics in the weekly comparison.
5. Hyperparameter tuning did not improve holdout performance. Despite better cross-validation RMSE, the tuned model's holdout RMSE and R² were slightly worse than the baseline's.
6. Evaluation level matters. Store-and-department performance does not automatically translate into equally strong performance on aggregated weekly sales.

## 💼 Business Impact and Recommendations

The aggregated XGBoost model achieved a **2.64% WAPE**, outperforming Prophet (3.87%) and SARIMAX (22.73%). This represents approximately **31.8% lower forecasting error than Prophet** and **88.4% lower than SARIMAX** on the evaluated weekly sales forecasts.

More accurate sales forecasts can support Walmart's operational and strategic decisions by helping teams:

- **Improve inventory management:** Anticipate demand more accurately to help reduce stockouts, overstocking, and inventory holding costs across stores and departments.
- **Plan staffing and resources:** align staffing plans with expected store-level demand.
- **Strengthen financial planning:** Develop more reliable sales projections for budgeting and revenue planning.
- **Optimize supply chains:** Improve replenishment planning and coordination across stores, warehouses, and suppliers.
- **Holiday preparation**: Plan inventory replenishment and distribution capacity ahead of periods with elevated sales.
- **Promotional planning**: Consider markdown periods alongside historical sales patterns when planning promotional activity.
- **Forecast monitoring**: Track prediction errors across stores, departments, and holiday periods to identify where model improvements are needed.
*These are potential business benefits inferred from forecast performance, not measured operational outcomes. The results demonstrate forecasting performance on the evaluated data; actual cost savings, inventory improvements, and revenue effects would require further business validation.*

Before operational deployment, I would validate performance over additional periods and investigate errors during high-demand weeks and across individual stores and departments.

## 📦 Tech Stack

Language:

\- Python

Libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `xgboost`
- `statsmodels`
- `prophet`

Techniques:

- Exploratory Data Analysis (EDA)
- Data integration and missing-value handling
- Time-based train/validation splitting
- Time-series cross-validation
- Temporal and sales-history feature engineering
- XGBoost regression
- Hyperparameter tuning
- SARIMAX and Prophet forecasting
- MAE, RMSE, MAPE, WAPE, and R² evaluation
- Interactive data visualization

## 📁 Repository Structure

Suggested repository layout:

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── features.csv
│   └── stores.csv
├── Walmart Store Sales Forecasting.ipynb
├── Walmart Store Sales Forecasting.html
├── requirements.txt
└── README.md
```

The raw datasets are not included in the README. Adjust the structure to match the files committed to the actual repository.

## 🚀 Usage / How to Run

Clone the repository:

```
git clone https://github.com/PreciousEbu/walmart-store-sales-forecasting.git
cd walmart-store-sales-forecasting
```

Replace the example repository URL with the actual URL once the project has been published.

Install the required packages:

```
pip install pandas numpy matplotlib plotly scikit-learn xgboost statsmodels prophet jupyter
```

Run the analysis:

1. Place the four CSV files in the `data/` directory, or update the paths in the notebook.
2. Launch Jupyter Notebook:

```
jupyter notebook
```

3\. Open the Walmart Store Sales Forecasting notebook and execute the cells in order.

The notebook uses local paths from the original development environment, so these paths must be updated before running it on another machine.

## 🤝 Acknowledgments & References

- Dataset Source: [Walmart Recruiting — Store Sales Forecasting (Kaggle)]([https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting?])
- XGBoost: [Official Documentation](https://xgboost.readthedocs.io/?)
- scikit-learn: [Official Documentation](https://scikit-learn.org/stable/?)
- Statsmodels: [Official Documentation](https://www.statsmodels.org/?)
- Prophet: [Official Documentation](https://facebook.github.io/prophet/?)

## 📄 License

No project license has been specified. Add a `LICENSE` file if you intend to grant others permission to reuse, modify, or redistribute the code.

## 👤 Author

Precious Ebu

- Portfolio: [https://preciousebu.github.io/](https://preciousebu.github.io/?)
- GitHub: [https://github.com/PreciousEbu](https://github.com/PreciousEbu?)
