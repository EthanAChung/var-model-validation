import numpy as np
import pandas as pd
import yfinance as yf

tickers = ["SPY", "QQQ", "IEF", "GLD", "USO"]

def load_prices():
    data = yf.download(tickers, start="2015-01-01", end="2025-12-31", auto_adjust=True)
    prices = data["Close"]
    prices = prices.ffill()
    prices.to_csv("data.csv")
    return prices

def calculate_returns(prices, weights):
    # weights is a dict like {"SPY": 0.5, "QQQ": 0.5}
    w = pd.Series(weights)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("Weights must sum to 1")
    if (w < 0).any():
        raise ValueError("Weights cannot be negative") 
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    returns = returns[w.index].dot(w)
    return returns

data = load_prices()
returns = calculate_returns(data, {"SPY": 0.3, "QQQ": 0.2, "IEF": 0.3, "GLD": 0.1, "USO": 0.1})
print(returns.head())
