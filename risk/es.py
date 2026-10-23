import numpy as np
import pandas as pd
from scipy.stats import norm, t
from arch import arch_model
from var import prep

def historical_es(returns, confidence=0.975):
    rets, alpha = prep(returns, confidence)
    threshold = np.percentile(rets, alpha * 100)
    return float(-rets[rets <= threshold].mean())

def parametric_es(returns, confidence=0.975):
    rets, alpha = prep(returns, confidence)
    z = norm.ppf(alpha)
    return float(-rets.mean() + rets.std(ddof=1) * norm.pdf(z) / alpha)

def monte_carlo_es(returns, confidence=0.975, num_sims=100_000,
                   dist="normal", seed=42):
    rets, alpha = prep(returns, confidence)
    rng = np.random.default_rng(seed)
    if dist == "normal":
        sims = rng.normal(rets.mean(), rets.std(ddof=1), num_sims)
    elif dist == "t":
        df, loc, scale = t.fit(rets)
        sims = t.rvs(df, loc=loc, scale=scale, size=num_sims, random_state=rng)
    else:
        raise ValueError("dist must be 'normal' or 't'.")
    threshold = np.percentile(sims, alpha * 100)
    return float(-sims[sims <= threshold].mean())

def garch_es(returns, confidence=0.975, dist="normal"):
    rets, alpha = prep(returns, confidence)
    rets = rets * 100
    res = arch_model(rets, mean="Constant", vol="GARCH", p=1, q=1,
                     dist=dist).fit(disp="off")
    sigma = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
    mu = res.params["mu"]
    if dist == "t":
        nu = res.params["nu"]
        q = t.ppf(alpha, nu)
        factor = (t.pdf(q, nu) / alpha * (nu + q**2) / (nu - 1)
                  * np.sqrt((nu - 2) / nu))
    elif dist == "normal":
        factor = norm.pdf(norm.ppf(alpha)) / alpha
    else:
        raise ValueError("dist must be 'normal' or 't'.")
    return float((-mu + sigma * factor) / 100)

def calculate_es(returns, confidence=0.975):
    return pd.Series({
        "Historical": historical_es(returns, confidence),
        "Parametric (normal)": parametric_es(returns, confidence),
        "Monte Carlo (normal)": monte_carlo_es(returns, confidence),
        "Monte Carlo (t)": monte_carlo_es(returns, confidence, dist="t"),
        "GARCH (normal)": garch_es(returns, confidence),
        "GARCH (t)": garch_es(returns, confidence, dist="t")}, 
        name="Expected Shortfall").to_frame()