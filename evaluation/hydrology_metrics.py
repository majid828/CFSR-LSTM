import numpy as np


_EPS = 1e-12


def _to_1d_float(x):
    """
    Convert input to a 1D float NumPy array.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _clean_pair(obs, pred):
    """
    Convert observed and predicted values to aligned 1D arrays and
    remove entries where either value is non-finite.

    Returns
    -------
    obs_clean : np.ndarray
    pred_clean : np.ndarray
    """
    o = _to_1d_float(obs)
    p = _to_1d_float(pred)

    if o.shape != p.shape:
        raise ValueError(
            f"obs and pred must have the same shape, got {o.shape} and {p.shape}"
        )

    mask = np.isfinite(o) & np.isfinite(p)

    o = o[mask]
    p = p[mask]

    if o.size == 0:
        raise ValueError("No finite paired observations remain after filtering.")

    return o, p


def mse(obs, pred):
    """
    Mean squared error.
    Lower is better.
    """
    o, p = _clean_pair(obs, pred)
    return float(np.mean((o - p) ** 2))


def rmse(obs, pred):
    """
    Root mean squared error.
    Lower is better.
    """
    return float(np.sqrt(mse(obs, pred)))


def mae(obs, pred):
    """
    Mean absolute error.
    Lower is better.
    """
    o, p = _clean_pair(obs, pred)
    return float(np.mean(np.abs(o - p)))


def bias(obs, pred):
    """
    Mean signed bias: mean(pred - obs).

    Positive value:
        systematic overprediction.

    Negative value:
        systematic underprediction.
    """
    o, p = _clean_pair(obs, pred)
    return float(np.mean(p - o))


def pbias(obs, pred):
    """
    Percent bias.

    PBIAS = 100 * sum(pred - obs) / sum(obs)

    Positive:
        total-flow overprediction.

    Negative:
        total-flow underprediction.

    Returns NaN if observed total is essentially zero.
    """
    o, p = _clean_pair(obs, pred)

    denom = np.sum(o)

    if abs(denom) <= _EPS:
        return float("nan")

    return float(100.0 * np.sum(p - o) / denom)


def correlation(obs, pred):
    """
    Pearson correlation coefficient.

    Returns NaN if either series has essentially zero variance.
    """
    o, p = _clean_pair(obs, pred)

    if np.std(o) <= _EPS or np.std(p) <= _EPS:
        return float("nan")

    return float(np.corrcoef(o, p)[0, 1])


def nse(obs, pred):
    """
    Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

    Interpretation:
        1     : perfect prediction
        0     : equivalent to predicting observed mean
        < 0   : worse than observed-mean predictor

    Returns NaN if observed variance is essentially zero.
    """
    o, p = _clean_pair(obs, pred)

    denom = np.sum((o - np.mean(o)) ** 2)

    if denom <= _EPS:
        return float("nan")

    return float(
        1.0 - np.sum((o - p) ** 2) / denom
    )


def kge(obs, pred):
    """
    Kling-Gupta Efficiency (2009-style formulation).

    KGE = 1 - sqrt(
        (r - 1)^2
        + (alpha - 1)^2
        + (beta - 1)^2
    )

    where
        r     = Pearson correlation
        alpha = std(pred) / std(obs)
        beta  = mean(pred) / mean(obs)

    Returns NaN when KGE is not well-defined.
    """
    o, p = _clean_pair(obs, pred)

    std_o = np.std(o)
    std_p = np.std(p)
    mean_o = np.mean(o)
    mean_p = np.mean(p)

    if std_o <= _EPS:
        return float("nan")

    if abs(mean_o) <= _EPS:
        return float("nan")

    if std_p <= _EPS:
        r = 0.0
    else:
        r = np.corrcoef(o, p)[0, 1]

    alpha = std_p / std_o
    beta = mean_p / mean_o

    return float(
        1.0
        - np.sqrt(
            (r - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def log_nse(obs, pred, eps=1e-6):
    """
    NSE computed in log-flow space.

    Useful for emphasizing low and moderate flows.

    Inputs are clipped to zero before adding eps because discharge
    should be nonnegative.

    Returns NaN if the transformed observed series has negligible variance.
    """
    o, p = _clean_pair(obs, pred)

    if eps <= 0:
        raise ValueError("eps must be positive.")

    o = np.maximum(o, 0.0)
    p = np.maximum(p, 0.0)

    log_o = np.log(o + eps)
    log_p = np.log(p + eps)

    return nse(log_o, log_p)


def low_flow_bias(obs, pred, quantile=0.2):
    """
    Percent bias restricted to observed low-flow conditions.

    Low-flow timesteps are defined by:
        obs <= quantile(obs, quantile)

    Returns
    -------
    percent bias

    Positive:
        low flows are overpredicted.

    Negative:
        low flows are underpredicted.

    Returns NaN when the selected observed low-flow total is nearly zero.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    o, p = _clean_pair(obs, pred)

    threshold = np.quantile(o, quantile)
    mask = o <= threshold

    if not np.any(mask):
        return float("nan")

    obs_low = o[mask]
    pred_low = p[mask]

    denom = np.sum(obs_low)

    if abs(denom) <= _EPS:
        return float("nan")

    return float(
        100.0 * np.sum(pred_low - obs_low) / denom
    )


def low_flow_rmse(obs, pred, quantile=0.2):
    """
    RMSE over observed low-flow conditions.

    Low-flow timesteps are selected using the observed-flow quantile.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    o, p = _clean_pair(obs, pred)

    threshold = np.quantile(o, quantile)
    mask = o <= threshold

    if not np.any(mask):
        return float("nan")

    return float(
        np.sqrt(np.mean((o[mask] - p[mask]) ** 2))
    )


def high_flow_rmse(obs, pred, quantile=0.95):
    """
    RMSE over observed high-flow conditions.

    Example:
        quantile=0.95 evaluates flows >= observed Q95 threshold.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    o, p = _clean_pair(obs, pred)

    threshold = np.quantile(o, quantile)
    mask = o >= threshold

    if not np.any(mask):
        return float("nan")

    return float(
        np.sqrt(np.mean((o[mask] - p[mask]) ** 2))
    )


def fdc_rmse(obs, pred):
    """
    Flow-duration-curve RMSE.

    Both observed and predicted flows are sorted independently in
    descending order before comparison.

    Lower is better.
    """
    o, p = _clean_pair(obs, pred)

    o_sorted = np.sort(o)[::-1]
    p_sorted = np.sort(p)[::-1]

    return float(
        np.sqrt(np.mean((o_sorted - p_sorted) ** 2))
    )


def evaluate_hydrology(obs, pred, low_quantile=0.2):
    """
    Convenience function returning the main hydrologic metrics.

    IMPORTANT:
    obs and pred should already represent a unique chronological
    time series. Do not pass flattened overlapping sequence windows.
    """
    return {
        "NSE": nse(obs, pred),
        "KGE": kge(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "Bias": bias(obs, pred),
        "PBIAS_percent": pbias(obs, pred),
        "Correlation": correlation(obs, pred),
        "logNSE": log_nse(obs, pred),
        "LowFlowBias_percent": low_flow_bias(
            obs,
            pred,
            quantile=low_quantile,
        ),
        "LowFlowRMSE": low_flow_rmse(
            obs,
            pred,
            quantile=low_quantile,
        ),
        "Q95_RMSE": high_flow_rmse(
            obs,
            pred,
            quantile=0.95,
        ),
        "Q99_RMSE": high_flow_rmse(
            obs,
            pred,
            quantile=0.99,
        ),
        "FDC_RMSE": fdc_rmse(obs, pred),
    }
