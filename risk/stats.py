import pandas as pd
from scipy import stats
from data.data import returns

def annual_mean(returns):
  daily_mean = returns.mean()
  annual_mean = daily_mean * 252
  return annual_mean

def volatility(returns):
  daily_vol = returns.std(ddof = 1)
  annual_vol = daily_vol * (252 ** 0.5)
  return annual_vol

def skewness(returns):
    skew = stats.skew(returns, bias=False)
    return skew

def excess_kurtosis(returns):
    kurtosis = stats.kurtosis(returns, bias=False)
    return kurtosis

def jarque_bera_test(returns):
    result = stats.jarque_bera(returns)
    pvalue = result.pvalue
    return pvalue

def summary_table(returns):
   return pd.Series({
      "Annual Mean": annual_mean(returns),
      "Volatility": volatility(returns),
      "Skewness": skewness(returns),
      "Kurtosis": excess_kurtosis(returns),
      "Jarque-Bera p-value": jarque_bera_test(returns)}, 
      name = "Value").to_frame()