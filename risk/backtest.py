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
    p = 1 - confidence
    pi = x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, df=1))

def christoffersen_test(breaches):
    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, df=1))

def backtest_summary(results, confidence=0.99):
    rows = {}
    for name, res in results.items():
        breaches = res["breach"]
        lr_uc, p_uc = kupiec_test(breaches, confidence)
        lr_ind, p_ind = christoffersen_test(breaches)
        rows[name] = {
            "Observations": len(breaches),
            "Breaches": int(breaches.sum()),
            "Expected breaches": len(breaches) * (1 - confidence),
            "Breach rate": f"{breaches.mean():.3f}",
            "Kupiec p-value": f"{p_uc:.3f}",
            "Christoffersen p-value": f"{p_ind:.3f}"}
    return pd.DataFrame(rows).T