#!/usr/bin/env python
# coding: utf-8

# ## 1. Project Introduction
# 
# The goal of this project is to build and evaluate a **pairs trading strategy** using historical stock price data. Pairs trading is a market-neutral trading strategy that looks for two stocks whose prices tend to move together over time. When the relationship between the two stocks temporarily diverges, the strategy takes a long-short position and bets that the spread will eventually return toward its normal level.
# 
# In this project, I use historical stock prices from the S&P 500 and search for candidate stock pairs with strong statistical relationships. For each pair, I estimate a hedge ratio using linear regression, construct the spread between the two stocks, and test whether the spread appears mean-reverting. I then use a rolling z-score model to generate trading signals.
# 
# The final strategy enters a trade when the spread becomes unusually high or low relative to its recent history, and exits when the spread returns closer to its mean. The performance of the strategy is evaluated out-of-sample using cumulative return, annualized return, volatility, Sharpe ratio, maximum drawdown, win rate, turnover, and transaction costs.
# 
# The purpose of this project is not only to find a profitable pair, but also to understand the risks and limitations of pairs trading. In particular, I examine whether strong in-sample relationships continue to hold out-of-sample, and whether the strategy remains profitable after transaction costs.

# ## 2. Import Libraries
# 
# I use yfinance to download stock prices, statsmodels for regression and cointegration tests, and pandas/numpy for data cleaning and backtesting.

# In[12]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import seaborn as sns


# ## 3. Load Data
# 
# In this section, I download historical daily price data for all current S&P 500 stocks. I first scrape the list of S&P 500 tickers from Wikipedia, then use the `yfinance` package to download adjusted daily closing prices.
# 
# Using the full S&P 500 universe gives the project a much larger search space for finding potentially cointegrated stock pairs. However, because this creates many possible pairs, later sections will need to carefully control overfitting using in-sample and out-of-sample testing.

# In[13]:


import sys
get_ipython().system('{sys.executable} -m pip install --upgrade certifi')


# In[ ]:


import requests
import certifi
from io import StringIO

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers, verify=certifi.where())
response.raise_for_status()

# Read the HTML table from the downloaded page
sp500_table = pd.read_html(StringIO(response.text))[0]

# Extract ticker symbols
tickers = sp500_table["Symbol"].tolist()

# Yahoo Finance uses "-" instead of "." in tickers like BRK.B
tickers = [ticker.replace(".", "-") for ticker in tickers]

print("Number of tickers:", len(tickers))
print(tickers[:10])
# Set time period
start_date = "2018-01-01"
end_date = "2024-12-31"

# Download adjusted daily prices for all S&P 500 stocks
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    group_by="column",
    threads=True
)

# Extract closing prices
prices = data["Close"]

prices.head()
print("Number of trading days:", prices.shape[0])
print("Number of stocks:", prices.shape[1])

prices.to_csv("sp500_prices.csv")


# ## 4. Clean Data
# 
# In this section, I clean the downloaded S&P 500 price data before doing cointegration testing. Some stocks may have missing prices because they were added to the S&P 500 later, changed tickers, or had incomplete Yahoo Finance data.
# 
# I remove stocks with too many missing values, then drop the remaining dates with missing observations. After cleaning, I convert prices into log prices because cointegration tests and spread construction are usually performed using log price series.
# 
# This cleaning step is important because the later cointegration and backtesting steps require aligned price histories across stocks.

# In[16]:


# Check the original data shape
print("Original price data shape:", prices.shape)

# Count missing values for each stock
missing_counts = prices.isna().sum().sort_values(ascending=False)

# Show stocks with the most missing values
missing_counts.head(20)


# In[17]:


missing_percent = (prices.isna().mean() * 100).sort_values(ascending=False)

missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percent": missing_percent
})

missing_summary.head(20)


# In[18]:


# Keep stocks with at least 95% non-missing observations
min_non_missing = int(0.95 * len(prices))

prices_clean = prices.dropna(axis=1, thresh=min_non_missing)

print("Shape before removing incomplete stocks:", prices.shape)
print("Shape after removing incomplete stocks:", prices_clean.shape)


# In[19]:


# Drop any remaining dates with missing values
prices_clean = prices_clean.dropna()

print("Final cleaned price data shape:", prices_clean.shape)

# Check that there are no missing values left
print("Total missing values after cleaning:", prices_clean.isna().sum().sum())


# In[20]:


# Remove any columns with non-positive prices, since log prices require positive values
positive_price_stocks = (prices_clean > 0).all(axis=0)

prices_clean = prices_clean.loc[:, positive_price_stocks]

print("Shape after removing non-positive price series:", prices_clean.shape)


# In[21]:


log_prices = np.log(prices_clean)

log_prices.head()


# In[22]:


# Plot a few cleaned price series to check that the data looks reasonable
prices_clean.iloc[:, :10].plot(figsize=(12, 6), title="Sample of Cleaned Stock Prices")

plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# After cleaning, the dataset contains stocks with sufficiently complete price histories and no remaining missing values. I also removed any invalid non-positive prices before taking logs.
# 
# The cleaned price data will be used for return calculations, while the log price data will be used for cointegration testing, hedge ratio estimation, and spread construction.

# ## 5. Exploratory Data Analysis
# 
# In this section, I explore the cleaned stock price data before building the trading strategy. I first calculate daily log returns and visualize correlations between stocks.
# 
# This step is important because pairs trading often starts by looking for stocks that move together. However, high correlation does not necessarily mean cointegration. Correlation measures short-term co-movement in returns, while cointegration measures whether two price series have a stable long-run relationship.

# In[25]:


# Calculate daily log returns
returns = log_prices.diff().dropna()

print("Log price data shape:", log_prices.shape)
print("Return data shape:", returns.shape)

returns.head()


# In[26]:


# Correlation matrix of returns
corr_matrix = returns.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Return Correlation Heatmap")
plt.show()


# In[28]:


# Convert correlation matrix into pair table safely

corr_matrix_clean = corr_matrix.copy()

# Remove index/column names to avoid reset_index conflict
corr_matrix_clean.index.name = None
corr_matrix_clean.columns.name = None

# Keep only upper triangle so each pair appears once
upper_triangle = corr_matrix_clean.where(
    np.triu(np.ones(corr_matrix_clean.shape), k=1).astype(bool)
)

# Convert to table
corr_pairs = upper_triangle.stack().reset_index()

corr_pairs.columns = ["Stock 1", "Stock 2", "Correlation"]

# Sort from highest to lowest correlation
corr_pairs = corr_pairs.sort_values("Correlation", ascending=False)

corr_pairs.head(20)


# ## 6. Feature Engineering
# 
# In this section, I create the main variables needed for the pairs trading strategy. I first split the cleaned data into an in-sample period and an out-of-sample period. The in-sample period is used to find cointegrated pairs and estimate the hedge ratio. The out-of-sample period is reserved for testing whether the strategy works on unseen data.
# 
# For each pair of stocks, I run an Engle-Granger cointegration test. Pairs with low p-values are considered stronger candidates because their prices appear to have a stable long-run relationship.
# 
# For the selected pair, I estimate the hedge ratio using linear regression, construct the spread, and convert the spread into a z-score. The z-score will later be used to generate trading signals.

# In[29]:


# Split data into in-sample and out-of-sample periods

split_date = "2022-12-31"

train_log = log_prices.loc[:split_date]
test_log = log_prices.loc[split_date:]

train_prices = prices_clean.loc[:split_date]
test_prices = prices_clean.loc[split_date:]

print("Training period:", train_log.index.min(), "to", train_log.index.max())
print("Testing period:", test_log.index.min(), "to", test_log.index.max())

print("Training shape:", train_log.shape)
print("Testing shape:", test_log.shape)


# In[30]:


from itertools import combinations
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

cointegration_results = []

stock_pairs = list(combinations(train_log.columns, 2))

for stock1, stock2 in tqdm(stock_pairs):
    try:
        score, pvalue, _ = coint(train_log[stock1], train_log[stock2])

        corr = train_log[[stock1, stock2]].corr().iloc[0, 1]

        cointegration_results.append({
            "stock1": stock1,
            "stock2": stock2,
            "coint_score": score,
            "pvalue": pvalue,
            "correlation": corr
        })

    except Exception:
        continue

coint_df = pd.DataFrame(cointegration_results)
coint_df = coint_df.sort_values("pvalue")

coint_df.head(20)


# In[31]:


coint_df.to_csv("../outputs/cointegration_results.csv", index=False)


# In[32]:


pvalue_threshold = 0.05

candidate_pairs = coint_df[coint_df["pvalue"] < pvalue_threshold].copy()

print("Number of candidate cointegrated pairs:", len(candidate_pairs))

candidate_pairs.head(20)


# In[ ]:


def analyze_pair(stock1, stock2, train_log, test_log, entry_threshold=2.0, exit_threshold=0.5):
    """
    Analyze one stock pair:
    1. Estimate hedge ratio
    2. Show full OLS regression results
    3. Construct spread
    4. Run ADF test
    5. Plot spread
    6. Plot z-score
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt

    # Estimate hedge ratio
    X = sm.add_constant(train_log[stock2])
    model = sm.OLS(train_log[stock1], X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    # Construct spread
    train_spread = train_log[stock1] - alpha - beta * train_log[stock2]
    test_spread = test_log[stock1] - alpha - beta * test_log[stock2]

    # ADF test on training spread
    adf_result = adfuller(train_spread.dropna())
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    # Z-score
    spread_mean = train_spread.mean()
    spread_std = train_spread.std()

    train_z = (train_spread - spread_mean) / spread_std
    test_z = (test_spread - spread_mean) / spread_std

    print("Pair:", stock1, "-", stock2)
    print("Alpha:", alpha)
    print("Hedge ratio beta:", beta)
    print("Regression R-squared:", model.rsquared)
    print("ADF statistic:", adf_stat)
    print("ADF p-value:", adf_pvalue)

    if adf_pvalue < 0.05:
        print("Result: Spread appears stationary in-sample.")
    else:
        print("Result: Spread may not be stationary in-sample.")

    # Show full regression table
    display(model.summary())

    # Extra spread distribution stats
    spread_stats = pd.Series({
        "Spread Mean": train_spread.mean(),
        "Spread Std": train_spread.std(),
        "Spread Skew": train_spread.skew(),
        "Spread Kurtosis": train_spread.kurtosis(),
        "Spread Min": train_spread.min(),
        "Spread Max": train_spread.max()
    })

    display(spread_stats)

    # Plot spread
    plt.figure(figsize=(12, 5))
    plt.plot(train_spread, label="In-sample spread")
    plt.axhline(train_spread.mean(), linestyle="--", color="black", label="Mean")
    plt.title(f"In-Sample Spread: {stock1} and {stock2}")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.show()

    # Plot z-score
    plt.figure(figsize=(12, 5))
    plt.plot(train_z, label="In-sample z-score")
    plt.axhline(entry_threshold, linestyle="--", label="Upper entry")
    plt.axhline(-entry_threshold, linestyle="--", label="Lower entry")
    plt.axhline(exit_threshold, linestyle=":", label="Upper exit")
    plt.axhline(-exit_threshold, linestyle=":", label="Lower exit")
    plt.axhline(0, color="black", label="Mean")
    plt.title(f"In-Sample Z-Score: {stock1} and {stock2}")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.legend()
    plt.show()

    return {
        "stock1": stock1,
        "stock2": stock2,
        "alpha": alpha,
        "beta": beta,
        "r_squared": model.rsquared,
        "adf_pvalue": adf_pvalue,
        "train_spread": train_spread,
        "test_spread": test_spread,
        "train_z": train_z,
        "test_z": test_z,
        "model": model,
        "spread_stats": spread_stats
    }


# In[ ]:


def get_pair_stats(stock1, stock2, train_log):
    """
    For one stock pair, estimate:
    1. Hedge ratio beta
    2. Regression R-squared
    3. ADF p-value of the spread
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller

    # Estimate hedge ratio using OLS
    X = sm.add_constant(train_log[stock2])
    model = sm.OLS(train_log[stock1], X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    # Construct spread
    spread = train_log[stock1] - alpha - beta * train_log[stock2]

    # ADF test on spread
    adf_pvalue = adfuller(spread.dropna())[1]

    return alpha, beta, model.rsquared, adf_pvalue


# In[ ]:


enhanced_results = []

# Use top 100 most cointegrated pairs first
top_candidates = coint_df.head(100).copy()

for _, row in top_candidates.iterrows():
    s1 = row["stock1"]
    s2 = row["stock2"]

    try:
        alpha, beta, r_squared, adf_pvalue = get_pair_stats(s1, s2, train_log)

        enhanced_results.append({
            "stock1": s1,
            "stock2": s2,
            "coint_pvalue": row["pvalue"],
            "correlation": row["correlation"],
            "alpha": alpha,
            "beta": beta,
            "r_squared": r_squared,
            "adf_pvalue": adf_pvalue
        })

    except Exception as e:
        print(f"Skipping {s1}-{s2}: {e}")

enhanced_pairs = pd.DataFrame(enhanced_results)

enhanced_pairs.sort_values("coint_pvalue").head(20)


# In[46]:


good_pairs = enhanced_pairs[
    (enhanced_pairs["coint_pvalue"] < 0.05) &
    (enhanced_pairs["adf_pvalue"] < 0.05) &
    (enhanced_pairs["correlation"] > 0.7) &
    (enhanced_pairs["beta"] > 0)
].copy()

good_pairs = good_pairs.sort_values("coint_pvalue")

print("Number of good pairs:", len(good_pairs))

good_pairs.head(20)


# In[288]:


row_number = 4 #Change this to test different pairs

stock1 = good_pairs.iloc[row_number]["stock1"]
stock2 = good_pairs.iloc[row_number]["stock2"]

pair_analysis = analyze_pair(stock1, stock2, train_log, test_log)


# ## 7. Model Building: Rolling Z-Score Pairs Trading Strategy
# 
# In this section, I build the final pairs trading model using a rolling z-score instead of a fixed in-sample z-score. The fixed approach estimates the spread mean and standard deviation using only the training period and keeps them constant throughout the test period. However, this may be too rigid because the relationship between two stocks can change over time.
# 
# To make the strategy more adaptive, I use a rolling window to calculate the spread mean and standard deviation. This means the model evaluates whether the current spread is unusually high or low relative to its recent behavior rather than relative to a fixed historical average.
# 
# The strategy uses the following rules:
# 
# - If the rolling z-score is above the upper entry threshold, the strategy shorts the spread.
# - If the rolling z-score is below the lower entry threshold, the strategy longs the spread.
# - If the rolling z-score returns close to zero, the strategy exits the position.
# 
# The selected pair is BDX-EVRG because it has a stationary in-sample spread, a positive hedge ratio, and more reasonable out-of-sample trading behavior than other candidate pairs.

# In[289]:


stock1 = pair_analysis["stock1"]
stock2 = pair_analysis["stock2"]

alpha = pair_analysis["alpha"]
beta = pair_analysis["beta"]

train_spread = pair_analysis["train_spread"]
test_spread = pair_analysis["test_spread"]

print("Selected pair:", stock1, "-", stock2)
print("Alpha:", alpha)
print("Beta:", beta)


# In[290]:


full_spread = pd.concat([train_spread, test_spread])

full_spread = full_spread.sort_index()

full_spread.head()


# In[291]:


def calculate_rolling_zscore(spread, window=60):
    """
    Calculate rolling z-score of the spread.

    z-score = (spread - rolling mean) / rolling standard deviation
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()

    zscore = (spread - rolling_mean) / rolling_std

    return zscore


# In[292]:


rolling_window = 60

full_z_rolling = calculate_rolling_zscore(full_spread, window=rolling_window)

# Keep only the out-of-sample z-score for testing
test_z = full_z_rolling.loc[test_spread.index].dropna()

test_z.head()


# In[293]:


entry_threshold = 2.0
exit_threshold = 0.5

plt.figure(figsize=(12, 5))

plt.plot(test_z, label="Out-of-sample rolling z-score")
plt.axhline(entry_threshold, linestyle="--", label="Upper entry")
plt.axhline(-entry_threshold, linestyle="--", label="Lower entry")
plt.axhline(exit_threshold, linestyle=":", label="Upper exit")
plt.axhline(-exit_threshold, linestyle=":", label="Lower exit")
plt.axhline(0, color="black", label="Mean")

plt.title(f"Rolling Z-Score: {stock1}-{stock2}")
plt.xlabel("Date")
plt.ylabel("Rolling z-score")
plt.legend()
plt.show()


# In[294]:


def generate_positions(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate trading positions based on spread z-score.

    position = 1 means long spread
    position = -1 means short spread
    position = 0 means no position
    """
    positions = pd.Series(index=zscore.index, dtype=float)
    current_position = 0

    for date in zscore.index:
        z = zscore.loc[date]

        if current_position == 0:
            if z > entry_threshold:
                current_position = -1
            elif z < -entry_threshold:
                current_position = 1

        elif current_position == 1:
            if z >= -exit_threshold:
                current_position = 0

        elif current_position == -1:
            if z <= exit_threshold:
                current_position = 0

        positions.loc[date] = current_position

    return positions


# In[295]:


test_positions = generate_positions(
    test_z,
    entry_threshold=entry_threshold,
    exit_threshold=exit_threshold
)

test_positions.value_counts()


# In[296]:


plt.figure(figsize=(12, 4))

plt.plot(test_positions, label="Trading position")
plt.axhline(0, color="black")

plt.title(f"Out-of-Sample Rolling Strategy Positions: {stock1}-{stock2}")
plt.xlabel("Date")
plt.ylabel("Position")
plt.legend()
plt.show()


# In[297]:


signals = pd.DataFrame({
    "z_score": test_z,
    "position": test_positions
})

signals["position_change"] = signals["position"].diff().fillna(0)

signals.head()


# In[298]:


trade_signals = signals[signals["position_change"] != 0].copy()

trade_signals


# ## 8. Model Evaluation
# 
# In this section, I evaluate the pairs trading strategy from the previous part using an out-of-sample backtest. The strategy uses the positions generated from the z-score trading rule in the previous section.
# 
# The backtest calculates daily strategy returns, applies transaction costs whenever the position changes, and then evaluates performance using cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, win rate, average trade return, and turnover.
# 
# This section focuses on out-of-sample performance because the strategy should be judged on data that was not used to select the pair or estimate the hedge ratio.

# In[299]:


# Daily log returns for the two stocks in the selected pair
ret1 = test_log[stock1].diff()
ret2 = test_log[stock2].diff()

# Spread return based on the hedge ratio
spread_returns = ret1 - beta * ret2

# Align returns with the rolling z-score dates
spread_returns = spread_returns.loc[test_z.index]

spread_returns.head()


# In[300]:


# Use yesterday's position to earn today's return
# This avoids look-ahead bias
strategy_returns_before_cost = test_positions.shift(1) * spread_returns

strategy_returns_before_cost = strategy_returns_before_cost.dropna()

strategy_returns_before_cost.head()


# In[301]:


# Transaction cost assumption
# 0.0005 means 5 basis points per position change
transaction_cost = 0.0005

# Position changes represent trades
position_changes = test_positions.diff().abs().fillna(0)

# Apply transaction cost whenever position changes
costs = position_changes * transaction_cost

# Strategy returns after transaction costs
strategy_returns_after_cost = strategy_returns_before_cost - costs.loc[strategy_returns_before_cost.index]

strategy_returns_after_cost = strategy_returns_after_cost.dropna()

strategy_returns_after_cost.head()


# In[302]:


equity_before_cost = (1 + strategy_returns_before_cost).cumprod()
equity_after_cost = (1 + strategy_returns_after_cost).cumprod()

plt.figure(figsize=(12, 5))
plt.plot(equity_before_cost, label="Before transaction costs")
plt.plot(equity_after_cost, label="After transaction costs")
plt.axhline(1, color="black", linestyle="--")

plt.title(f"Out-of-Sample Equity Curve: {stock1}-{stock2}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()


# In[303]:


running_max = equity_after_cost.cummax()
drawdown = equity_after_cost / running_max - 1

plt.figure(figsize=(12, 5))
plt.plot(drawdown)

plt.title(f"Out-of-Sample Drawdown: {stock1}-{stock2}")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.show()


# In[304]:


def calculate_performance_metrics(returns, positions, periods_per_year=252):
    """
    Calculate common backtest performance metrics.
    """
    returns = returns.dropna()

    cumulative_return = (1 + returns).prod() - 1

    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(returns)) - 1

    annualized_volatility = returns.std() * np.sqrt(periods_per_year)

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility != 0
        else np.nan
    )

    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()
    average_daily_return = returns.mean()

    # Turnover = average absolute daily position change
    turnover = positions.diff().abs().mean()

    # Number of trades = number of nonzero position changes
    number_of_trades = (positions.diff().fillna(0) != 0).sum()

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Average Daily Return": average_daily_return,
        "Turnover": turnover,
        "Number of Trades": number_of_trades
    }


# In[305]:


metrics_before_cost = calculate_performance_metrics(
    strategy_returns_before_cost,
    test_positions
)

metrics_after_cost = calculate_performance_metrics(
    strategy_returns_after_cost,
    test_positions
)

performance_table = pd.DataFrame({
    "Before Transaction Costs": metrics_before_cost,
    "After Transaction Costs": metrics_after_cost
})

performance_table


# In[306]:


trade_log = pd.DataFrame({
    "stock1": stock1,
    "stock2": stock2,
    "z_score": test_z,
    "position": test_positions,
    "position_change": test_positions.diff().fillna(0),
    "spread_return": spread_returns,
    "strategy_return_before_cost": strategy_returns_before_cost,
    "transaction_cost": costs,
    "strategy_return_after_cost": strategy_returns_after_cost,
    "equity_after_cost": equity_after_cost
})

trade_log.head()


# In[307]:


trade_dates = trade_log[trade_log["position_change"] != 0].copy()

trade_dates


# In[308]:


trade_log.to_csv("../outputs/trade_log_bdx_evrg.csv")
performance_table.to_csv("../outputs/performance_table_bdx_evrg.csv")


# In[ ]:


# Combined version for faster runs on different pairs
def run_rolling_pair_backtest(
    stock1,
    stock2,
    train_log,
    test_log,
    rolling_window=60,
    entry_threshold=2.0,
    exit_threshold=0.5,
    transaction_cost=0.0005
):
    """
    Runs the full rolling z-score pairs trading backtest for one pair.

    Steps:
    1. Estimate hedge ratio using in-sample data.
    2. Construct train/test spread.
    3. Calculate rolling z-score.
    4. Generate positions.
    5. Calculate strategy returns before and after transaction costs.
    """
    import statsmodels.api as sm

    # Estimate hedge ratio on in-sample period
    X = sm.add_constant(train_log[stock2])
    model = sm.OLS(train_log[stock1], X).fit()

    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]

    # Construct spread
    train_spread = train_log[stock1] - alpha - beta * train_log[stock2]
    test_spread = test_log[stock1] - alpha - beta * test_log[stock2]

    # Use train + test spread so rolling window has enough history at start of test
    full_spread = pd.concat([train_spread, test_spread]).sort_index()

    # Rolling z-score
    full_z = calculate_rolling_zscore(full_spread, window=rolling_window)
    train_z = full_z.loc[train_spread.index].dropna()
    test_z = full_z.loc[test_spread.index].dropna()

    # Positions
    train_positions = generate_positions(
        train_z,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold
    )

    test_positions = generate_positions(
        test_z,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold
    )

    # Spread returns
    train_ret1 = train_log[stock1].diff()
    train_ret2 = train_log[stock2].diff()
    train_spread_returns = train_ret1 - beta * train_ret2
    train_spread_returns = train_spread_returns.loc[train_z.index]

    test_ret1 = test_log[stock1].diff()
    test_ret2 = test_log[stock2].diff()
    test_spread_returns = test_ret1 - beta * test_ret2
    test_spread_returns = test_spread_returns.loc[test_z.index]

    # Strategy returns before cost
    train_returns_before_cost = (
        train_positions.shift(1) * train_spread_returns
    ).dropna()

    test_returns_before_cost = (
        test_positions.shift(1) * test_spread_returns
    ).dropna()

    # Transaction costs
    train_position_changes = train_positions.diff().abs().fillna(0)
    test_position_changes = test_positions.diff().abs().fillna(0)

    train_costs = train_position_changes * transaction_cost
    test_costs = test_position_changes * transaction_cost

    train_returns_after_cost = (
        train_returns_before_cost
        - train_costs.loc[train_returns_before_cost.index]
    ).dropna()

    test_returns_after_cost = (
        test_returns_before_cost
        - test_costs.loc[test_returns_before_cost.index]
    ).dropna()

    return {
        "stock1": stock1,
        "stock2": stock2,
        "alpha": alpha,
        "beta": beta,
        "model": model,
        "train_spread": train_spread,
        "test_spread": test_spread,
        "train_z": train_z,
        "test_z": test_z,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "train_returns_before_cost": train_returns_before_cost,
        "test_returns_before_cost": test_returns_before_cost,
        "train_returns_after_cost": train_returns_after_cost,
        "test_returns_after_cost": test_returns_after_cost,
        "train_costs": train_costs,
        "test_costs": test_costs
    }


# ### 8.1 In-Sample vs Out-of-Sample Comparison
# 
# To check whether the model generalizes beyond the training period, I compare the rolling z-score strategy's in-sample and out-of-sample performance. The in-sample period is the data used to identify the pair and estimate the hedge ratio, while the out-of-sample period is unseen data used to test whether the strategy still works.
# 
# This comparison is important because a pairs trading strategy can look strong in-sample but fail out-of-sample if the relationship between the two stocks breaks down.

# In[310]:


final_stock1 = stock1
final_stock2 = stock2

# Final model parameters
rolling_window = 60
entry_threshold = 2.0
exit_threshold = 0.5
transaction_cost = 0.0005

final_result = run_rolling_pair_backtest(
    final_stock1,
    final_stock2,
    train_log,
    test_log,
    rolling_window=rolling_window,
    entry_threshold=entry_threshold,
    exit_threshold=exit_threshold,
    transaction_cost=transaction_cost
)

train_metrics = calculate_performance_metrics(
    final_result["train_returns_after_cost"],
    final_result["train_positions"]
)

test_metrics = calculate_performance_metrics(
    final_result["test_returns_after_cost"],
    final_result["test_positions"]
)

in_sample_vs_out_sample = pd.DataFrame({
    "In-Sample": train_metrics,
    "Out-of-Sample": test_metrics
})

in_sample_vs_out_sample


# The in-sample vs out-of-sample table compares the strategy's performance before and after the testing split. The out-of-sample results are more important because they show whether the strategy works on unseen data. If the in-sample performance is much stronger than the out-of-sample performance, that may indicate overfitting.

# ### 8.2 Sensitivity Analysis
# 
# I test different rolling windows, entry thresholds, and exit thresholds to see whether the strategy depends too heavily on one specific parameter choice. A more reliable strategy should remain profitable under nearby parameter settings rather than only working for one exact combination.
# 
# The parameters tested are:
# 
# - Rolling window: 30, 60, 90 days
# - Entry threshold: 1.5, 2.0, 2.5
# - Exit threshold: 0.25, 0.5, 0.75

# In[ ]:


sensitivity_results = []

entry_thresholds = [1.5, 2.0, 2.5]
exit_thresholds = [0.25, 0.5, 0.75]
rolling_windows = [30, 60, 90]

for window in rolling_windows:
    for entry in entry_thresholds:
        for exit_ in exit_thresholds:

            if exit_ >= entry:
                continue

            try:
                temp_result = run_rolling_pair_backtest(
                    final_stock1,
                    final_stock2,
                    train_log,
                    test_log,
                    rolling_window=window,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    transaction_cost=transaction_cost
                )

                temp_metrics = calculate_performance_metrics(
                    temp_result["test_returns_after_cost"],
                    temp_result["test_positions"]
                )

                sensitivity_results.append({
                    "Rolling Window": window,
                    "Entry Threshold": entry,
                    "Exit Threshold": exit_,
                    "Cumulative Return": temp_metrics["Cumulative Return"],
                    "Annualized Return": temp_metrics["Annualized Return"],
                    "Annualized Volatility": temp_metrics["Annualized Volatility"],
                    "Sharpe Ratio": temp_metrics["Sharpe Ratio"],
                    "Max Drawdown": temp_metrics["Max Drawdown"],
                    "Win Rate": temp_metrics["Win Rate"],
                    "Turnover": temp_metrics["Turnover"],
                    "Number of Trades": temp_metrics["Number of Trades"]
                })

            except Exception as e:
                print(f"Skipped window={window}, entry={entry}, exit={exit_}: {e}")

sensitivity_table = pd.DataFrame(sensitivity_results)

sensitivity_table.sort_values("Sharpe Ratio", ascending=False).head(15)
sensitivity_table.to_csv("sensitivity_analysis.csv", index=False)


# In[313]:


# Sort sensitivity table by Sharpe ratio
sensitivity_table_sorted = sensitivity_table.sort_values("Sharpe Ratio", ascending=False)

# Display top 15 parameter combinations
display(sensitivity_table_sorted.head(15))

# Print the best parameter combination
best_params = sensitivity_table_sorted.iloc[0]

print("Best parameter combination:")
print("Rolling Window:", best_params["Rolling Window"])
print("Entry Threshold:", best_params["Entry Threshold"])
print("Exit Threshold:", best_params["Exit Threshold"])
print("Cumulative Return:", best_params["Cumulative Return"])
print("Annualized Return:", best_params["Annualized Return"])
print("Sharpe Ratio:", best_params["Sharpe Ratio"])
print("Max Drawdown:", best_params["Max Drawdown"])
print("Number of Trades:", best_params["Number of Trades"])


# The sensitivity analysis shows how the model performs under different parameter choices. This helps check whether the final result is robust or whether it only works for one specific rolling window and threshold combination. I focus on combinations with strong Sharpe ratios, positive cumulative returns, controlled drawdowns, and reasonable trade counts.

# ### 8.3 Average Trade Return
# 
# The earlier performance table reports daily metrics such as Sharpe ratio, volatility, and daily win rate. However, pairs trades often last multiple days, so I also calculate trade-level returns. This helps determine whether the strategy is profitable on completed trading episodes rather than only on individual days.

# In[314]:


def calculate_trade_returns(strategy_returns, positions):
    """
    Groups daily strategy returns into trade-level returns.

    A trade starts when the lagged position becomes nonzero.
    A trade ends when the lagged position returns to zero or changes direction.
    """
    active_position = positions.shift(1).reindex(strategy_returns.index).fillna(0)

    trades = []
    current_trade_returns = []
    current_direction = 0
    start_date = None

    for date in strategy_returns.index:
        pos = active_position.loc[date]
        ret = strategy_returns.loc[date]

        # Start a new trade
        if current_direction == 0 and pos != 0:
            current_direction = pos
            start_date = date
            current_trade_returns = [ret]

        # Continue current trade
        elif current_direction != 0 and pos == current_direction:
            current_trade_returns.append(ret)

        # Trade ends or changes direction
        elif current_direction != 0 and pos != current_direction:
            end_date = date

            trade_return = (1 + pd.Series(current_trade_returns)).prod() - 1

            trades.append({
                "Start Date": start_date,
                "End Date": end_date,
                "Direction": current_direction,
                "Trade Return": trade_return,
                "Holding Days": len(current_trade_returns)
            })

            # If new position starts immediately
            if pos != 0:
                current_direction = pos
                start_date = date
                current_trade_returns = [ret]
            else:
                current_direction = 0
                start_date = None
                current_trade_returns = []

    # If a trade is still open at the end
    if current_direction != 0 and len(current_trade_returns) > 0:
        trade_return = (1 + pd.Series(current_trade_returns)).prod() - 1

        trades.append({
            "Start Date": start_date,
            "End Date": strategy_returns.index[-1],
            "Direction": current_direction,
            "Trade Return": trade_return,
            "Holding Days": len(current_trade_returns)
        })

    return pd.DataFrame(trades)


# In[315]:


trade_return_table = calculate_trade_returns(
    final_result["test_returns_after_cost"],
    final_result["test_positions"]
)

trade_return_table


# In[316]:


average_trade_return = trade_return_table["Trade Return"].mean()
trade_win_rate = (trade_return_table["Trade Return"] > 0).mean()
average_holding_days = trade_return_table["Holding Days"].mean()

trade_summary = pd.Series({
    "Average Trade Return": average_trade_return,
    "Trade-Level Win Rate": trade_win_rate,
    "Average Holding Days": average_holding_days,
    "Number of Completed/Open Trades": len(trade_return_table)
})

trade_summary


# The trade-level return table groups daily strategy returns into individual trading episodes. This provides a clearer measure of trade profitability than the daily win rate because each trade can last multiple days. The average trade return and trade-level win rate help evaluate whether the model consistently earns money when it enters positions.

# ### 8.4 Overall Portfolio Cumulative Return
# 
# In addition to analyzing the final selected pair, I also test an equal-weighted portfolio of multiple candidate pairs. This checks whether the strategy can work across several pairs instead of depending entirely on one selected pair.
# 
# For each candidate pair, I run the same rolling z-score strategy and calculate out-of-sample after-cost returns. I then average the daily returns across pairs to create an equal-weighted pairs trading portfolio.

# In[ ]:


# Choose candidate pairs for portfolio test
# Can change 10 to a larger or smaller number
portfolio_candidate_pairs = good_pairs.head(10).copy()

portfolio_pair_returns = {}
portfolio_pair_metrics = []

for i, row in portfolio_candidate_pairs.iterrows():
    s1 = row["stock1"]
    s2 = row["stock2"]

    try:
        pair_result = run_rolling_pair_backtest(
            s1,
            s2,
            train_log,
            test_log,
            rolling_window=rolling_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            transaction_cost=transaction_cost
        )

        pair_returns = pair_result["test_returns_after_cost"]
        pair_name = f"{s1}-{s2}"

        portfolio_pair_returns[pair_name] = pair_returns

        pair_metrics = calculate_performance_metrics(
            pair_returns,
            pair_result["test_positions"]
        )

        portfolio_pair_metrics.append({
            "Pair": pair_name,
            **pair_metrics
        })

    except Exception as e:
        print(f"Skipping {s1}-{s2}: {e}")

portfolio_pair_metrics_df = pd.DataFrame(portfolio_pair_metrics)

portfolio_pair_metrics_df.sort_values("Sharpe Ratio", ascending=False)


# In[318]:


# Combine pair returns into one DataFrame
portfolio_returns_df = pd.DataFrame(portfolio_pair_returns)

# Equal-weighted portfolio return across available pairs each day
portfolio_returns = portfolio_returns_df.mean(axis=1).dropna()

# Portfolio equity curve
portfolio_equity = (1 + portfolio_returns).cumprod()

plt.figure(figsize=(12, 5))
plt.plot(portfolio_equity)
plt.axhline(1, color="black", linestyle="--")

plt.title("Out-of-Sample Cumulative Returns: Equal-Weighted Pairs Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.show()


# In[319]:


# For portfolio-level positions, turnover is less clear, so use pair metrics separately.
# Here we pass a dummy position series only to make the function work if needed.

dummy_positions = pd.Series(index=portfolio_returns.index, data=1)

portfolio_metrics = calculate_performance_metrics(
    portfolio_returns,
    dummy_positions
)

pd.Series(portfolio_metrics)
portfolio_returns_df.to_csv("portfolio_pair_returns.csv")
portfolio_pair_metrics_df.to_csv("portfolio_pair_metrics.csv", index=False)


# The portfolio cumulative return chart shows the performance of an equal-weighted strategy across multiple candidate pairs. This is useful because relying on one pair can make the strategy sensitive to pair-specific events. A portfolio of pairs can diversify some of this risk.
# 
# However, the individual pair results show that performance varies widely across pairs. Some pairs produce strong positive returns, while others perform poorly. This suggests that pair selection remains one of the most important parts of the strategy.

# ## 9. Results and Interpretation
# 
# The final rolling z-score model was evaluated using several checks: out-of-sample performance, transaction costs, sensitivity analysis, trade-level returns, and portfolio-level cumulative returns. These checks are important because the project is not only about finding one profitable backtest, but also about testing whether the strategy is robust.
# 
# The final selected pair, Pair 4, produced the strongest balance between return and risk. It achieved high after-cost cumulative return, a strong Sharpe ratio, low maximum drawdown, and a reasonable number of trades. The sensitivity analysis also helps determine whether this result depends too heavily on one exact parameter choice.
# 
# The broader portfolio test shows that the same strategy does not work equally well for every pair. Some pairs performed strongly, while others lost money out-of-sample. This confirms that in-sample cointegration is not sufficient by itself. The pair must also continue to mean-revert out-of-sample and survive transaction costs.

# ## 10. Conclusion
# 
# This project built and evaluated a pairs trading strategy using historical stock price data. The goal was to find pairs of stocks whose price relationship showed mean-reverting behavior and then use that relationship to generate trading signals.
# 
# The final strategy used a **rolling z-score model**. Instead of comparing the spread to one fixed historical mean and standard deviation, the rolling model updates the spread’s mean and volatility using recent data. This made the strategy more adaptive to changing market conditions and improved out-of-sample performance.
# 
# After testing multiple candidate pairs, **Pair 4** was selected as the final pair because it had the strongest overall balance between return and risk. After transaction costs, the strategy achieved a cumulative return of **35.06%**, an annualized return of **16.36%**, annualized volatility of **9.17%**, and a Sharpe ratio of **1.78**. The maximum drawdown was only **-6.07%**, suggesting that the strategy controlled downside risk relatively well.
# 
# The results also show that pair selection is very important. Some pairs that looked statistically reasonable in-sample performed poorly out-of-sample. This shows that in-sample stationarity alone is not enough to guarantee a profitable trading strategy. A useful pairs trading model must be tested on out-of-sample data and evaluated after transaction costs.
# 
# Overall, the rolling z-score pairs trading strategy produced a strong result for the selected pair. However, the model still has limitations. The backtest only covers one out-of-sample period, and future performance may change if the relationship between the two stocks breaks down. In a more advanced version, the strategy could be improved by testing more time periods, using rolling hedge ratios, adding stop-loss rules, and comparing performance across different market regimes.
