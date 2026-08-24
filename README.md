##### **Predicting Next-Day Stock Direction Using Price-Volume Features**



###### **A Real-World Data Project - Finance Domain**



**Overview**



This project investigates whether short-term stock price direction can be predicted using historical price and volume data alone. It follows an end-to-end data science pipeline: real market data collection, exploratory data analysis (EDA), feature engineering, predictive modeling, and strategy backtesting - the kind of workflow used in quantitative finance research.



**Core question:** Can lagged returns, moving averages, volatility, volume, and momentum indicators predict whether a stock's price will go up or down the next trading day — and if a model based on these signals is used to trade, does it actually outperform simply buying and holding?



**Dataset**



* **Source:** Yahoo Finance, accessed via the `yfinance` Python library
* **Tickers:** AAPL, MSFT, GOOGL, AMZN, JPM, XOM, SPY (7 liquid large-cap stocks/ETF spanning tech, finance, energy, and a broad market benchmark)
* **Period:** January 1, 2015 – August 23, 2026 (\~11.5 years of daily data, 2,926 trading days)
* **Fields used:** Adjusted daily Close price, daily Volume



This dataset was chosen over fundamental/accounting data, cryptocurrency, or static Kaggle datasets because daily OHLCV time series is the standard data format used in real quantitative finance research and trading strategy development.



**Methodology**



1\. Exploratory Data Analysis

* Normalized price performance across all 7 tickers (growth of $1 invested)
* Daily returns computed as percentage change in close price
* Distribution analysis of returns: skewness and kurtosis, to check for "fat tails" (a well-known property of financial returns vs. a normal distribution)
* Rolling 21-day annualized volatility, to visualize volatility clustering (calm periods vs. crisis periods)
* Cross-ticker correlation matrix of daily returns
* Augmented Dickey-Fuller (ADF) stationarity test - comparing raw prices vs. returns



2\. Feature Engineering

Built on SPY (S\&P 500 ETF) as the primary prediction target, using only information available up to the prior trading day:

* Lagged returns (1-day, 2-day, 5-day)
* Short and long moving average deviations (5-day, 20-day)
* Rolling volatility (10-day standard deviation of returns)
* Volume z-score (how unusual today's volume is vs. the trailing 20-day average)
* 14-day Relative Strength Index (RSI), a standard momentum indicator



3\. Predictive Modeling

* Target: Binary classification - will next-day return be positive (1) or negative (0)?
* Model: Random Forest Classifier (200 trees, max depth 5)
* Train/test split: Chronological 80/20 split (not shuffled - critical for time series, to avoid leaking future information into training)
* Features standardized using StandardScaler before training
* Two model variants tested: default class weighting, and `class\_weight="balanced"` to address class imbalance (more "Up" days than "Down" days historically)



4\. Strategy Backtest

* Simulated a simple long-only strategy: go long when the model predicts "Up," stay flat otherwise
* Compared cumulative returns against a passive buy-and-hold benchmark
* Calculated Sharpe ratio and maximum drawdown for both strategy and benchmark
* Adjusted returns for estimated transaction costs (5 basis points per trade) to produce a "net of costs" performance line



**Key Findings**



|**Metric**|**Value**|
|-|-|
|Model accuracy (default weighting)|55.5%|
|Model accuracy (class-balanced)|49.1%|
|Naive baseline (always predict "Up")|57.6%|
|Strategy Sharpe ratio|\~1.16|
|Buy \& Hold Sharpe ratio|\~1.25|
|Strategy max drawdown|\~-17.5%|
|Buy \& Hold max drawdown|\~-18.8%|
|ADF p-value, raw prices|0.998 (non-stationary)|
|ADF p-value, returns|\~3.5e-30 (stationary)|
|SPY daily returns kurtosis|14.06 (strongly fat-tailed vs. normal distribution)|





**1. Prices are non-stationary; returns are stationary.**

The ADF test confirms a foundational result in quantitative finance: raw price series show no evidence of stationarity (p ≈ 0.998), while daily returns are strongly stationary (p ≈ 0). This is why virtually all quantitative models work with returns rather than raw prices.



**2. Financial returns exhibit fat tails.**

Kurtosis of 14.06 (far above the value of 3 expected for a normal distribution) confirms that extreme return days occur far more often than a normal distribution would predict - visible directly in the volatility spikes during 2020 (COVID) and other periods in the rolling volatility chart.



**3. Volatility clusters over time.**

Rather than being random and evenly spread, large price swings tend to cluster together (calm markets stay calm, turbulent markets stay turbulent) — clearly visible in the rolling 21-day annualized volatility chart, with the sharpest spike during the 2020 market crash.



**4. The model does not meaningfully beat a naive baseline.**

The default model achieved 55.5% accuracy on next-day direction — below the "always predict Up" baseline of 57.6%. This is a legitimate and expected finding, not a failure: it is broadly consistent with weak-form market efficiency, which holds that past price and volume data alone should not reliably predict future returns.



**5. Class-balancing hurt raw accuracy but changed model behavior.**

Forcing the model to weight the minority "Down" class more heavily reduced overall accuracy (49.1%) because the model became less confident in its dominant "Up" predictions without becoming meaningfully better at identifying "Down" days. This highlights a common trade-off in imbalanced classification: balancing techniques don't automatically improve a model if the underlying signal for the minority class is weak.



**6. The trading strategy underperformed passive investing, but with a smaller drawdown.**

Despite the modest predictive edge, the model-driven strategy produced a lower Sharpe ratio and lower cumulative return than simply buying and holding SPY over the test period. Interestingly, its maximum drawdown was slightly smaller (-17.5% vs. -18.8%), suggesting the model avoided being fully invested during some of the worst days - a partial, if incomplete, risk-management benefit.



**7. Transaction costs matter.**

After accounting for an estimated 5 basis points per trade, the net-of-costs strategy return was further reduced from the gross figure - a reminder that even a strategy with a real (if small) edge can be eroded by trading frictions in practice.



**Conclusions**



This project demonstrates that next-day stock direction is very difficult to predict using price and volume features alone - a result consistent with market efficiency theory rather than a shortcoming of the modeling approach. The value of this project lies less in producing a profitable trading signal and more in demonstrating a rigorous, honest quantitative research process: proper time-series train/test methodology, awareness of data stationarity, evaluation against a naive baseline, and risk-adjusted performance metrics (Sharpe ratio, drawdown) rather than accuracy alone.



**Limitations**



* Only price/volume-derived features were used; no fundamental, macroeconomic, or sentiment data was incorporated
* The backtest is simplified (no slippage modeling, no position sizing, single-asset long-only strategy)
* Random Forest was the only model architecture tested; results may differ with other approaches (logistic regression, gradient boosting, LSTM)
* Prediction horizon was fixed at one trading day; results may differ over longer horizons



**Future Work**



* Incorporate sentiment data (news, social media) as additional features
* Test longer prediction horizons (weekly, monthly) where signal may be stronger
* Explore regime-switching or volatility-prediction targets instead of raw direction
* Apply the same pipeline across the full 7-ticker basket rather than SPY alone, to test cross-sectional strategies
* Test alternative models (gradient boosting, logistic regression with regularization) as a robustness check



**Tools \& Libraries**



yfinance | pandas | numpy | matplotlib | seaborn | scikit-learn | statsmodels



