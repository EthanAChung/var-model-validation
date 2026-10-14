import pandas as pd
from scipy import stats

def annual_mean(returns):
  daily_mean = returns.mean()
  return daily_mean * 252

def volatility(returns):
  daily_vol = returns.std(ddof = 1)
  return daily_vol * (252 ** 0.5)

def skewness(returns):
    return stats.skew(returns, bias=False)

def excess_kurtosis(returns):
    return stats.kurtosis(returns, bias=False)

def jarque_bera_test(returns):
    result = stats.jarque_bera(returns)
    return result.pvalue

def summary_table(returns):
   return pd.Series({
      "Annual Mean": annual_mean(returns),
      "Volatility": volatility(returns),
      "Skewness": skewness(returns),
      "Kurtosis": excess_kurtosis(returns),
      "Jarque-Bera p-value": jarque_bera_test(returns)}, 
      name = "Value").to_frame()