# Walmart Store Sales Forecasting

align="center">align="center">

`<img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white" alt="Pandas">`{=html}
`<img src="https://img.shields.io/badge/Scikit--learn-Model%20Evaluation-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">`{=html}
`<img src="https://img.shields.io/badge/XGBoost-Forecasting-EC6B23" alt="XGBoost">`{=html}
`<img src="https://img.shields.io/badge/Plotly-Interactive%20Visuals-3F4F75?logo=plotly&logoColor=white" alt="Plotly">`{=html}
`<img src="https://img.shields.io/badge/Status-Completed-success" alt="Project status">`{=html}


## Project Overview

This project explores historical Walmart sales data and develops
time-series forecasting models to estimate weekly sales. It combines
exploratory data analysis, holiday and promotion analysis, feature
engineering, time-aware model validation, hyperparameter tuning, and
model comparison.

The goal is to produce forecasts that can inform retail planning---not
simply to fit historical data. More reliable weekly sales estimates can
help retailers plan inventory, staffing, promotions, and distribution
capacity around demand fluctuations.

## Business Problem

Retail sales vary with seasonality, holidays, store characteristics,
department-level demand, promotions, and wider economic conditions.
Forecasting errors can contribute to overstocking, stockouts,
inefficient staffing, and poorly timed promotional activity.

**Objective:** forecast Walmart weekly sales using historical sales and
relevant contextual variables, then compare statistical time-series
methods with a feature-based machine-learning approach.

## Dataset

The notebook uses four CSV files commonly associated with the Walmart
Store Sales forecasting dataset:

  --------------------------------------------------------------------------
  File                                Rows × columns Purpose
  --------------------- ---------------------------- -----------------------
  `train.csv`                            421,570 × 5 Historical weekly sales
                                                     by store and department

  `test.csv`                             115,064 × 4 Store/department/date
                                                     records for forecasting

  `features.csv`                          8,190 × 12 Holiday indicators,
                                                     temperature, fuel
                                                     price, markdowns, CPI,
                                                     and unemployment

  `stores.csv`                                45 × 3 Store metadata,
                                                     including store type
                                                     and size
  --------------------------------------------------------------------------

The training target is `Weekly_Sales`. The data covers store- and
department-level sales across multiple weeks. The notebook's displayed
date range for the modeling series runs from **5 February 2010 to 26
October 2012**.

> **Data note:** the notebook reads local CSV paths from the original
> development environment. Update those paths or place the CSV files in
> the project's data directory before rerunning it. The raw dataset is
> not included in this README.

## Tools and Technologies

-   **Python** --- analysis and model development
-   **Pandas, NumPy** --- data manipulation and feature preparation
-   **Matplotlib, Plotly** --- exploratory charts and interactive
    visualizations
-   **scikit-learn** --- evaluation metrics, time-series
    cross-validation, and randomized hyperparameter search
-   **XGBoost** --- regression forecasting with engineered features
-   **SARIMAX and Prophet** --- statistical forecasting benchmarks

## Workflow and Methodology

### 1. Data loading and integration

Loaded the sales, store, and feature datasets, parsed dates, and joined
the sources to create a modeling table containing sales history and
contextual variables.

### 2. Data quality and missing values

Reviewed missing values after merging. The five markdown variables
(`MarkDown1`--`MarkDown5`) contained substantial missingness. The
notebook treated missing markdown values as no recorded promotion and
filled them with zero. This assumption should be revisited if missing
values can also mean unreported promotional activity.

### 3. Exploratory data analysis

Explored weekly sales over time and compared sales patterns by: -
holiday versus non-holiday weeks; - store type and store size; - store
and department; - markdown and economic context.

The notebook findings indicate that sales show pronounced peaks around
holiday periods and that store size and promotional markdowns are
relevant factors in sales variation.

### 4. Time-based validation

For the store/department machine-learning workflow, the notebook used a
chronological split at **1 January 2012**: - Training period: 294,132
rows - Validation period: 127,438 rows

This preserves temporal ordering and reduces the risk of training on
future observations when evaluating past forecasts.

### 5. Feature engineering and model development

Built a feature-based XGBoost regressor using store, department,
holiday, weather, fuel-price, economic, markdown, and
store-characteristic information, together with sales-history features.
A baseline XGBoost model was evaluated, followed by time-series
cross-validation and randomized hyperparameter tuning.

The tuning search tested 15 parameter configurations over five
time-series folds (75 fits). The selected configuration reported by the
notebook was:

  Parameter              Selected value
  -------------------- ----------------
  `n_estimators`                    800
  `max_depth`                         6
  `learning_rate`                  0.05
  `subsample`                       0.8
  `colsample_bytree`                0.6

The notebook also evaluated **SARIMAX** and **Prophet** on an aggregated
weekly sales series to compare traditional time-series approaches
against the aggregated output of the XGBoost workflow.

### 6. Evaluation

The notebook reports: - **MAE** --- average absolute prediction error; -
**RMSE** --- penalizes larger errors more heavily; - **MAPE** --- mean
absolute percentage error; - **WAPE** --- total absolute error relative
to total actual sales; - **R²** --- variance explained relative to a
mean-prediction baseline.

Lower MAE, RMSE, MAPE, and WAPE indicate smaller errors; higher R² is
generally better. Percentage metrics can become unstable when actual
sales are zero or close to zero, so WAPE is particularly useful as a
complementary measure.

## Model Performance Comparison

The following results are from the notebook's **aggregated weekly-sales
comparison table**. XGBoost predictions were aggregated from
store/department forecasts to the weekly total before comparison with
SARIMAX and Prophet.

  ---------------------------------------------------------------------------------------
  Model                       RMSE             MAE       MAPE\*       WAPE\*           R²
  ---------------- --------------- --------------- ------------ ------------ ------------
  **XGBoost          **1,752,574**   **1,227,584**     **2.61**     **2.64**    **0.482**
  (aggregated)**                                                             

  Prophet                2,423,342       1,800,569         3.85         3.87        0.010

  SARIMAX               10,893,740      10,570,583        22.72        22.73      -19.004
  ---------------------------------------------------------------------------------------

\*The notebook reports MAPE and WAPE as numeric values interpreted as
percentages.

### What the comparison shows

-   **XGBoost ranked first** on the aggregated comparison, with the
    lowest RMSE, MAE, MAPE, and WAPE of the three models.
-   Its reported **WAPE of 2.64%** was lower than Prophet's 3.87% and
    SARIMAX's 22.73%.
-   XGBoost's aggregated **R² was 0.482**, indicating that it explained
    about 48.2% of the variation in the aggregated weekly target under
    this evaluation.
-   Prophet was the second-best model by error metrics, but its R² of
    0.010 indicated limited explanatory power in this setup.
-   SARIMAX performed poorly in this comparison: its R² was negative,
    meaning it performed worse than a simple mean-based prediction
    baseline on the evaluated observations.

### Store/department-level XGBoost validation

The tuned XGBoost model also produced the following results on the
store/department validation data:

  Metric                           Tuned XGBoost result
  ------------------------------ ----------------------
  MAE                                          1,477.33
  RMSE                                         3,093.43
  R²                                              0.980
  TimeSeriesSplit best CV RMSE                 4,598.08

The notebook reports a baseline XGBoost validation RMSE of **2,977.27**
and R² of **0.982**, compared with the tuned model's RMSE of
**3,093.43** and R² of **0.980**. Thus, the selected hyperparameters
improved the best cross-validation RMSE relative to the initial average
CV RMSE (4,885.50 to 4,598.08), but did **not** improve the final
validation RMSE versus the notebook's baseline. This is an important
distinction: tuning did not improve every holdout metric.

**Comparison caveat:** the store/department validation metrics and
aggregated weekly comparison metrics are calculated at different levels
and have different target scales. They should not be compared directly.
The notebook's reported MAPE values for the store/department model are
extremely large because percentage error is unstable with zero or
near-zero actual sales; they are not presented here as a meaningful
performance headline.

## Key Findings

1.  **Holiday periods matter.** Weekly sales show peaks around major
    holiday periods, supporting explicit holiday-aware planning.
2.  **Store characteristics matter.** Store size and store type are
    useful context for understanding differences in sales levels.
3.  **Promotional activity is relevant.** Markdown variables provide
    information about promotional periods and should be considered when
    forecasting demand.
4.  **Feature-based machine learning outperformed the tested statistical
    benchmarks** in the notebook's aggregated weekly comparison.
5.  **Evaluation level matters.** Strong store/department R² does not
    necessarily translate to equally strong performance on aggregated
    weekly totals; both should be checked against the intended business
    use.

## Business Impact and Recommendations

This analysis demonstrates how a forecasting workflow can support retail
operations. The notebook does not quantify realized savings, revenue
uplift, or inventory reductions, so the benefits below are **potential
operational applications**, not measured business outcomes.

-   **Inventory planning:** use forecasts to identify expected demand
    peaks and allocate stock across stores and departments.
-   **Holiday readiness:** prepare inventory and replenishment plans
    ahead of high-demand holiday weeks.
-   **Workforce scheduling:** align staffing plans with expected
    store-level demand.
-   **Promotion planning:** assess how markdown periods align with sales
    patterns and use them as forecast inputs where reliable data is
    available.
-   **Supply chain coordination:** use weekly forecasts to inform
    replenishment timing and distribution capacity.
-   **Model monitoring:** track forecast errors by store, department,
    holiday status, and store type; retrain or recalibrate when
    performance deteriorates.

### Recommended next steps

1.  Evaluate the model on a genuinely unseen future period and compare
    it with simple seasonal-naïve baselines.
2.  Report error metrics by store and department, not only at the
    aggregate level.
3.  Inspect residuals around holidays and promotion periods.
4.  Reassess missing-markdown handling and test alternative imputation
    assumptions.
5.  Use a zero-safe percentage metric or prioritize WAPE/MAE where
    actual sales can be zero.
6.  Connect forecasts to inventory and staffing decisions, then measure
    business outcomes through a controlled pilot.

## Visualizations

The notebook includes: - total weekly sales trend; - holiday versus
non-holiday sales trend; - store-type sales comparisons; -
forecast-versus-actual charts for SARIMAX and Prophet; - aggregated
model comparison and error analysis; - interactive Plotly sales and
holiday dashboards.

If you publish screenshots, place them in a `screenshots/` directory and
link them here, for example:

``` markdown
![Weekly sales trend](screenshots/weekly-sales-trend.png)
![Forecast comparison](screenshots/model-comparison.png)
```

## Repository Structure

A suggested structure for publishing the project:

``` text
walmart-store-sales-forecasting/
├── README.md
├── Walmart Store Sales Forecasting.ipynb
├── data/                 # Optional; keep raw data out of Git if licensing restricts it
├── screenshots/          # Optional charts for this README
└── requirements.txt      # Optional environment dependencies
```

The uploaded notebook is an HTML export. If that is the version stored
in your repository, change the notebook filename above to match the
actual filename.

## How to Run

1.  Clone or download the repository.
2.  Install the libraries used by the notebook.
3.  Obtain the dataset files and place them where the notebook expects
    them, or update the file paths.
4.  Open the notebook in Jupyter Notebook or JupyterLab and run the
    cells in order.

Example setup:

``` bash
pip install pandas numpy matplotlib plotly scikit-learn xgboost statsmodels prophet jupyter
```

The exact environment may require additional version-specific setup for
Prophet and its dependencies.

## Limitations

-   The results are from the notebook's validation design and should not
    be interpreted as proof of production performance.
-   The aggregated model comparison covers the evaluated weekly
    validation observations; it does not establish performance for every
    store, department, or future period.
-   Forecasting accuracy can vary substantially by store, department,
    and holiday period.
-   Missing markdown values were treated as zero, an assumption that may
    not always hold.
-   Percentage error metrics are unreliable when actual sales are zero
    or near zero.
-   No realized financial or operational impact was measured in this
    project.

## Conclusion

The notebook compares SARIMAX, Prophet, and XGBoost for Walmart weekly
sales forecasting. In the aggregated weekly comparison, **XGBoost
delivered the strongest error metrics**, with RMSE of approximately 1.75
million, MAE of approximately 1.23 million, WAPE of 2.64%, and R² of
0.482. The analysis also highlights holiday-related sales peaks and the
relevance of store characteristics and promotional markdowns.

The practical takeaway is to use forecasts as decision support for
inventory, staffing, promotions, and replenishment---while continuing to
validate performance at the level where operational decisions are made.

## Author

**Precious Ebu**

-   Portfolio: https://preciousebu.github.io/
-   GitHub: https://github.com/PreciousEbu
