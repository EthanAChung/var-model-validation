import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

def rolling_var(returns, var_func, confidence=0.99, window=250, step=1):
    rets = pd.Series(returns).astype(float).dropna()
    var_values = pd.Series(np.nan, index=rets.index)
    for i in range(window, len(rets), step):
        var_values.iloc[i:i + step] = var_func(rets.iloc[i - window:i], confidence)
    var_values = var_values.dropna()
    actual = rets.loc[var_values.index]
    return pd.DataFrame({
        "return": actual,
        "var": var_values,
        "breach": actual < -var_values,
    })

def kupiec_test(breaches, confidence=0.99):
    b = np.asarray(breaches, dtype=bool)
    n, x = len(b), int(b.sum())
    p = 1 - confidence          # expected breach rate
    pi = x / n                  # observed breach rate
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, df=1))

def backtest_summary(results, confidence=0.99):
    rows = {}
    for name, res in results.items():
        breaches = res["breach"]
        lr_uc, p_uc = kupiec_test(breaches, confidence)
        rows[name] = {
            "Observations": len(breaches),
            "Breaches": int(breaches.sum()),
            "Expected breaches": len(breaches) * (1 - confidence),
            "Breach rate": breaches.mean(),
            "Kupiec p-value": p_uc}
    return pd.DataFrame(rows).T