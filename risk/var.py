import numpy as np
import pandas as pd
from scipy.stats import norm, t
from arch import arch_model

def prep(returns, confidence):
    if not 0.8 < confidence < 1:
        raise ValueError(
            "confidence must be between 0.8 and 1, "
            f"e.g. 0.95 or 0.99 (got {confidence}).")
    rets = np.squeeze(np.asarray(returns, dtype=float))
    if rets.ndim != 1:
        raise ValueError(
            "returns must be a single series (e.g. portfolio returns), "
            f"not an array of shape {rets.shape}.")
    rets = rets[~np.isnan(rets)]
    return rets, 1 - confidence

def historical_var(returns, confidence=0.99):
    rets, alpha = prep(returns, confidence)
    return float(-np.percentile(rets, alpha * 100))

def parametric_var(returns, confidence=0.99):
    rets, alpha = prep(returns, confidence)
    return float(-(rets.mean() + norm.ppf(alpha) * rets.std(ddof=1)))

def monte_carlo_var(returns, confidence=0.99, num_sims=100_000, dist="normal", seed=42):
    rets, alpha = prep(returns, confidence)
    rng = np.random.default_rng(seed)
    if dist == "normal":
        sims = rng.normal(rets.mean(), rets.std(ddof=1), num_sims)
    elif dist == "t":
        df, loc, scale = t.fit(rets)
        sims = t.rvs(df, loc=loc, scale=scale, size=num_sims, random_state=rng)
    else:
        raise ValueError("distribution must be normal or t")
    return float(-np.percentile(sims, alpha * 100))

def garch_var(returns, confidence=0.99, dist="normal"):
    rets, alpha = prep(returns, confidence)
    rets = rets * 100
    res = arch_model(rets, mean="Constant", vol="GARCH", p=1, q=1,
                     dist=dist).fit(disp="off")
    sigma = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
    mu = res.params["mu"]
    if dist == "t": 
        nu = res.params["nu"]
        q = t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
    elif dist == "normal":
        q = norm.ppf(alpha)
    else:
        raise ValueError("dist must be 'normal' or 't'.")
    return float(-(mu + q * sigma) / 100)

def calculate_vars(returns, confidence=0.99):
    """All VaR estimates for one return series."""
    return pd.Series({
        "Historical": historical_var(returns, confidence),
        "Parametric (normal)": parametric_var(returns, confidence),
        "Monte Carlo (normal)": monte_carlo_var(returns, confidence),
        "Monte Carlo (t)": monte_carlo_var(returns, confidence, dist="t"),
        "GARCH (normal)": garch_var(returns, confidence),
        "GARCH (t)": garch_var(returns, confidence, dist="t")}, 
        name = "VaR Estimates").to_frame()