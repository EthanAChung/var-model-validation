import numpy as np
import pandas as pd
import yfinance as yf

def load_prices(tickers, start="2015-01-01", end="2025-12-31"):
    data = yf.download(tickers, start, end, auto_adjust=True)
    prices = data["Close"]
    prices = prices.ffill()
    return prices

def calculate_returns(prices, weights):
    w = pd.Series(weights)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("Weights must sum to 1")
    if (w < 0).any():
        raise ValueError("Weights cannot be negative") 
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    returns = returns[w.index].dot(w)
    return returns