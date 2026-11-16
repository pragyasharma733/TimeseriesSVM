# Forecasting NYSE Trading Volume with Random Forests

> Note: despite the repository name, this project uses a **random forest**, not an SVM. Consider renaming the repo to `nyse-volume-forecasting`.

One-step-ahead forecasting of daily NYSE trading volume using lagged market features.

## Question
Can we predict tomorrow's (log) trading volume from the last 5 days of volume, Dow Jones returns and market volatility, and does a machine learning model beat a naive "tomorrow = today" forecast?

## Data
`NYSE.csv` comes from the R package `ISLR2`. It contains 6,051 trading days, from December 1962 to December 1986.

| Variable | Meaning |
|---|---|
| `DJ_return` | Daily Dow Jones return |
| `log_volume` | Log of trading volume (detrended), which is the target |
| `log_volatility` | Log of absolute daily price movement |

## Approach
1. Turned the series into a supervised learning table: 15 features, made up of 5 lags each of volume, return and volatility, predicting volume at *t+1*.
2. **Time-series cross-validation** (`TimeSeriesSplit`, 3 folds). Each fold's test block is split 50/50 into validation and test, so there is no look-ahead leakage.
3. Tuned `n_estimators` ∈ {100, 500, 750} and cost-complexity pruning `ccp_alpha` ∈ {10⁰ … 10⁻⁹}.
4. Compared against a naive persistence baseline.
5. Examined feature importances.

## Results
| Model | RMSE |
|---|---|
| Naive baseline (cₜ₊₁ = cₜ) | 0.189 |
| Random forest (500 trees, α = 10⁻⁵) | 0.187 |

- Best validation RMSE was 0.160.
- The most recent day's volume (cₜ) accounts for **53%** of feature importance.

## How to run
```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook "Time Series Data.ipynb"
```

## Limitations and next steps
- The baseline RMSE is computed over the whole series while the random forest is scored on the last fold's test block. Compare both on the same period for a fair test.
- The random forest beats the baseline by only ~1%. That is an honest and interesting finding, and worth stating clearly.
- Add day-of-week features (already in the data), plus linear/ARIMA and gradient-boosting comparisons.
- Plot predicted vs actual volume over time.
