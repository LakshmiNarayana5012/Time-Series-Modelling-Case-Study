# Forecasting German Electricity Demand

## About this case study

This notebook is my time-series case study on German electricity demand. The main aim was to see whether more advanced forecasting methods could improve on simple benchmark forecasts when predicting the final two years of the available data.

I started with hourly demand data, explored the main patterns in the series, and then built the models in stages. The notebook includes:

- mean, naïve, seasonal-naïve and drift forecasts;
- a weekly SARIMA model;
- a SARIMAX model using Berlin temperature;
- a Gradient Boosting regression model;
- an hourly LSTM model with a small hyperparameter search.

The important comparison throughout the project is against the seasonal-naïve model. German electricity use has a strong yearly pattern, so a complex model is only useful if it performs better than simply repeating the demand from the same week one year earlier.

## Files used

The main notebook is:

```text
Time_Series_Modelling_Case_Study(4).ipynb
```

The electricity dataset must be saved in the same folder as the notebook:

```text
time_series_60min_singleindex.csv
```

The notebook uses the following two columns from that file:

```text
utc_timestamp
DE_load_actual_entsoe_transparency
```

The electricity data come from the Open Power System Data time-series package:

<https://data.open-power-system-data.org/time_series/>

Berlin temperature data are downloaded inside the notebook from the Open-Meteo historical weather API:

<https://archive-api.open-meteo.com/v1/archive>

An internet connection is therefore needed when Part 4 is run.

## Data used in the analysis

The notebook keeps German electricity-demand observations from 1 January 2015 onward.

The saved notebook run contains:

```text
Hourly observations: 50,400
Daily observations: 2,100
Weekly observations: 301
Start: 1 January 2015
End: 30 September 2020
Missing German load values: 0
```

Hourly demand is measured in megawatts. For the weekly models, the hourly values are averaged into Sunday-ending weeks using `W-SUN`.

The final 104 weeks are used as the weekly test set. For the LSTM, the final 17,520 hourly observations, equal to two 365-day years, are used for testing.

## What the notebook does

### Part 1: data preparation and exploration

The first section loads the German demand series and creates daily and weekly averages. It then examines the data using:

- daily and weekly time plots;
- average demand by hour;
- average demand by weekday;
- average demand by month and year;
- a 12-week rolling mean and standard deviation;
- STL decomposition with a seasonal period of 52 weeks;
- ACF and PACF plots;
- ADF and KPSS stationarity tests.

The STL result gives a seasonal-strength value of about `0.871`, which confirms that seasonality is a major part of the series.

The stationarity tests are applied to the original, first-differenced, seasonally differenced, and combined-differenced series. These results are used to support the differencing choices considered later in the SARIMA model.

### Part 2: benchmark forecasts

Four simple weekly forecasts are compared over the final 104 weeks:

1. **Mean forecast** – uses the average of the training data.
2. **Naïve forecast** – repeats the final training observation.
3. **Seasonal-naïve forecast** – repeats the value from 52 weeks earlier.
4. **Drift forecast** – continues the average change between the first and last training observations.

The notebook calculates MAE, RMSE, MASE and bias for each method.

Results from the saved run are:

| Model | MAE (MW) | RMSE (MW) | MASE | Bias (MW) |
|---|---:|---:|---:|---:|
| Seasonal naïve | 2,318.521 | 3,006.761 | 1.732 | 1,731.751 |
| Naïve | 3,783.203 | 4,459.109 | 2.827 | -882.480 |
| Mean | 3,788.833 | 4,397.300 | 2.831 | 481.006 |
| Drift | 4,339.891 | 5,117.957 | 3.243 | 1,006.802 |

Seasonal naïve is the strongest benchmark in the current run.

The benchmark section creates:

```text
output/benchmark_forecast_comparison.png
output/benchmark_forecasts.csv
output/benchmark_metrics.csv
```

### Part 3: SARIMA modelling

The SARIMA section uses a chronological training and test split with 197 training weeks and 104 testing weeks.

The notebook loops through every required non-seasonal combination of:

```text
p = 0 to 6
d = 0 to 2
q = 0 to 6
```

This gives 147 candidate models. AIC is used to select the best non-seasonal order.

The saved run selected:

```text
Order: (2, 1, 6)
Seasonal order: (0, 1, 0, 52)
AIC: 2428.232
```

The final model is checked using:

- a residual time plot;
- a residual ACF;
- a residual histogram;
- a Q-Q plot;
- Ljung-Box tests at lags 10, 20 and 52;
- a two-year forecast with 95% confidence intervals.

The Ljung-Box p-values in the saved run are all above 0.05, which suggests that no strong residual autocorrelation remains at those lags.

The test results are:

```text
RMSE: 4,652.240 MW
MAE:  3,834.350 MW
Bias: 3,697.876 MW
```

The SARIMA model therefore did not beat the seasonal-naïve benchmark in this run.

### Part 4: SARIMAX with temperature

The notebook downloads daily mean temperature for Berlin and converts it to weekly averages using the same Sunday-ending weekly frequency as the electricity data.

Temperature is then added to a SARIMAX model as an exogenous variable. The final 104 weeks are forecast using the observed temperature values from the test period.

This is a **conditional forecast**, not a fully operational forecast. In a real forecasting system, the future observed temperatures would not be available. Archived weather forecasts or simulated weather forecast errors would be needed for a fair operational test.

The current saved output reports:

```text
RMSE: 22,537.706 MW
MAE:  16,985.744 MW
Bias: -16,898.273 MW
```

The result file is saved as:

```text
sarimax_temperature_forecast.csv
```

### Part 5: Gradient Boosting

The feature-based model uses `GradientBoostingRegressor`.

Its input features include:

- weekly mean temperature;
- heating-degree and cooling-degree variables;
- sine and cosine terms for week-of-year;
- load lags of 1, 2, 4, 13 and 52 weeks;
- rolling means over 4, 13 and 52 weeks;
- rolling standard deviations over 4 and 13 weeks.

The load lags and rolling values are based only on earlier observations. During the test forecast, predicted demand is added back into the history instead of using the actual test demand. This keeps the forecast recursive and avoids target leakage.

The model used in the saved run has:

```text
n_estimators = 300
learning_rate = 0.03
max_depth = 2
min_samples_leaf = 5
loss = "huber"
random_state = 42
```

Its test performance is:

```text
MAE:  2,572.576 MW
RMSE: 3,432.984 MW
MASE: 1.922
Bias: 2,079.209 MW
```

Gradient Boosting is the closest advanced weekly model to seasonal naïve, but its RMSE is still higher in the current run.

The notebook also plots feature importance to show which inputs the fitted model relies on most heavily.

### Part 6: hourly LSTM

The LSTM section returns to the hourly electricity-demand series.

The model uses:

- the previous seven days as input (`168` hours);
- six cyclical calendar variables for hour, weekday and day of year;
- one scaled load feature;
- a 24-hour output horizon.

The final 90 days of the training period are held out for validation. The two-year test set is kept untouched during tuning.

Four candidate network designs are compared:

| Units | Layers | Dropout | Learning rate |
|---:|---:|---:|---:|
| 32 | 1 | 0.0 | 0.001 |
| 64 | 1 | 0.2 | 0.001 |
| 64 | 2 | 0.2 | 0.0005 |
| 96 | 2 | 0.3 | 0.0005 |

The best saved validation result came from:

```text
32 units
1 LSTM layer
0.0 dropout
0.001 learning rate
20 epochs
Validation MSE: 0.017616
```

The final model predicts one day at a time. Each 24-hour prediction is added to the model history before the next day is forecast. Actual test-period load is not used during this recursive process.

The LSTM section calculates MAE, RMSE, MAPE, sMAPE, bias and R-squared. It also creates a daily-average plot of the full two-year forecast and an hourly plot of the first four forecast weeks.

The following files are saved when the section finishes:

```text
lstm_hourly_two_year_forecast.csv
lstm_hourly_metrics.csv
lstm_hyperparameter_results.csv
german_electricity_lstm.keras
```

## Software needed

The notebook was written in Python and uses:

```text
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
requests
tensorflow
jupyter
```

The packages can be installed with:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn requests tensorflow jupyter
```

TensorFlow can run on a CPU, but the LSTM section will take longer without a GPU.

## How to run the notebook

1. Put the notebook and `time_series_60min_singleindex.csv` in the same folder.
2. Open a terminal in that folder.
3. Install the required packages.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Time_Series_Modelling_Case_Study(4).ipynb`.
6. Restart the kernel.
7. Run the cells from the beginning in order.
8. Keep an internet connection active for the Open-Meteo temperature request.
9. Check that all forecast files and plots are created before using the results in the written report.

The SARIMA grid search and LSTM training are the slowest sections.

## How data leakage is controlled

I used a chronological split rather than a random split because future observations should not influence the past.

The main leakage controls in the notebook are:

- all test observations occur after the training observations;
- load lags are created with `shift`;
- rolling statistics use earlier demand values;
- the LSTM scaler is fitted using training data only;
- the LSTM validation period comes from the end of the training set;
- recursive Gradient Boosting and LSTM forecasts use previous predictions rather than actual test loads;
- observed future temperature is clearly treated as conditional information.

## Main finding

The main result from the saved notebook is that greater model complexity did not automatically produce a better forecast.

Seasonal naïve has the lowest verified weekly RMSE:

```text
Seasonal naïve RMSE: 3,006.761 MW
Gradient Boosting RMSE: 3,432.984 MW
SARIMA RMSE: 4,652.240 MW
```

This makes sense because German electricity demand contains a strong annual pattern. The test period also includes the unusual fall in electricity use during 2020, which is difficult for a model trained only on earlier years to predict.

The practical conclusion is that seasonal naïve should remain the reference model. An advanced model should only replace it after showing a clear and stable improvement on the same test period.

## Important checks before final submission

The notebook runs and produces results, but the following points should be checked before treating every model result as final:

1. The seasonal SARIMA search cell currently repeats `(0, 1, 0, 52)` several times. The non-seasonal `p`, `d` and `q` grid is complete, but distinct seasonal `P`, `D` and `Q` candidates should be added if full seasonal tuning is required.

2. The SARIMAX temperature cell currently uses the variables `order` and `seasonal_order`. These may contain values left over from a previous loop. It should use:

   ```python
   order=best_order
   seasonal_order=best_seasonal_order
   ```

   The model should then be rerun and the SARIMAX metrics updated.

3. Gradient Boosting currently uses one chosen parameter set. A time-series validation search could be added if additional hyperparameter tuning is required.

4. The LSTM metrics are calculated and saved, but they are not printed in the saved notebook output. The contents of `lstm_hourly_metrics.csv` should be copied into the final results table.

5. For a fair comparison with the weekly models, the hourly LSTM forecast should also be aggregated to weekly averages and evaluated over the matching weekly dates.

6. Any placeholder values or figures in the report should be replaced with the final rerun outputs.

## Possible improvements

There are several ways the work could be extended:

- use rolling-origin cross-validation instead of relying on one train/test split;
- include separate temperature-only, holiday-only and combined SARIMAX comparisons;
- use temperature from several German cities instead of Berlin alone;
- evaluate models using archived weather forecasts rather than observed future temperature;
- tune Gradient Boosting with a time-series validation procedure;
- compare the LSTM with hourly seasonal-naïve baselines;
- train the LSTM with several random seeds;
- add prediction intervals to the tree and neural-network forecasts;
- test an ensemble of seasonal naïve, SARIMAX and Gradient Boosting.

## References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
- Friedman, J. H. (2001). “Greedy function approximation: A gradient boosting machine.” *The Annals of Statistics*, 29(5), 1189–1232.
- Hochreiter, S. and Schmidhuber, J. (1997). “Long short-term memory.” *Neural Computation*, 9(8), 1735–1780.
- Hyndman, R. J. and Koehler, A. B. (2006). “Another look at measures of forecast accuracy.” *International Journal of Forecasting*, 22(4), 679–688.
- Marino, D. L., Amarasinghe, K. and Manic, M. (2016). “Building energy load forecasting using deep neural networks.” *Proceedings of IECON 2016*.
- Wiese, F. et al. (2019). “Open Power System Data – Frictionless data for electricity system modelling.” *Applied Energy*, 236, 401–409.
