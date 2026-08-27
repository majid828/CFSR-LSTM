import numpy as np


_EPS = 1e-12


def _to_1d(x):
    return np.asarray(x).reshape(-1)


def _clean_inputs(obs, pred, event_id):
    """
    Convert inputs to aligned 1D arrays and remove entries where
    obs or pred is non-finite.

    event_id values are preserved for the retained positions.
    """
    o = np.asarray(obs, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    e = np.asarray(event_id).reshape(-1)

    if not (o.shape == p.shape == e.shape):
        raise ValueError(
            f"obs, pred, and event_id must have the same shape, "
            f"got {o.shape}, {p.shape}, {e.shape}"
        )

    mask = np.isfinite(o) & np.isfinite(p)

    o = o[mask]
    p = p[mask]
    e = e[mask]

    if o.size == 0:
        raise ValueError("No valid paired observations remain after filtering.")

    return o, p, e


def event_peak_errors(obs, pred, event_id, eps=1e-6):
    """
    Compute event-by-event peak magnitude diagnostics.

    Parameters
    ----------
    obs : array-like
        Observed discharge.
    pred : array-like
        Predicted discharge.
    event_id : array-like
        Event label at each time step.
        Negative IDs are ignored.
    eps : float
        Small value for relative-error denominator.

    Returns
    -------
    list of dict
        Each dictionary contains:
        - event
        - obs_peak
        - pred_peak
        - peak_error
        - abs_peak_error
        - relative_error_pct
        - signed_relative_error_pct
        - underpredicted
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    o, p, e = _clean_inputs(obs, pred, event_id)

    out = []

    for eid in np.unique(e):
        if eid < 0:
            continue

        mask = e == eid

        if not np.any(mask):
            continue

        obs_event = o[mask]
        pred_event = p[mask]

        obs_peak = float(np.max(obs_event))
        pred_peak = float(np.max(pred_event))

        peak_error = pred_peak - obs_peak
        abs_peak_error = abs(peak_error)

        denom = max(abs(obs_peak), eps)

        rel_error = 100.0 * abs_peak_error / denom
        signed_rel_error = 100.0 * peak_error / denom

        out.append({
            "event": int(eid),
            "obs_peak": obs_peak,
            "pred_peak": pred_peak,
            "peak_error": float(peak_error),
            "abs_peak_error": float(abs_peak_error),
            "relative_error_pct": float(rel_error),
            "signed_relative_error_pct": float(signed_rel_error),
            "underpredicted": bool(pred_peak < obs_peak),
        })

    return out


def peak_rmse(obs, pred, event_id):
    """
    RMSE of event peak magnitudes.

    Lower is better.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    errors = np.array(
        [x["pred_peak"] - x["obs_peak"] for x in events],
        dtype=float,
    )

    return float(np.sqrt(np.mean(errors ** 2)))


def peak_mae(obs, pred, event_id):
    """
    Mean absolute error of event peak magnitudes.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    errors = np.array(
        [x["pred_peak"] - x["obs_peak"] for x in events],
        dtype=float,
    )

    return float(np.mean(np.abs(errors)))


def mean_relative_peak_error(obs, pred, event_id):
    """
    Mean absolute relative event-peak error in percent.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    return float(
        np.mean([x["relative_error_pct"] for x in events])
    )


def median_relative_peak_error(obs, pred, event_id):
    """
    Median absolute relative event-peak error in percent.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    return float(
        np.median([x["relative_error_pct"] for x in events])
    )


def relative_peak_bias(obs, pred, event_id):
    """
    Mean signed relative event-peak bias in percent.

    Positive:
        peaks are overpredicted.

    Negative:
        peaks are underpredicted.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    return float(
        np.mean([x["signed_relative_error_pct"] for x in events])
    )


def quantile_rmse(obs, pred, q=0.95):
    """
    RMSE over time steps where observed flow is above
    the observed q-quantile threshold.

    Example:
        q=0.95 -> high-flow RMSE over observed Q95+ flows.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be between 0 and 1.")

    o = np.asarray(obs, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)

    if o.shape != p.shape:
        raise ValueError(
            f"obs and pred must have the same shape, got {o.shape} and {p.shape}"
        )

    mask_valid = np.isfinite(o) & np.isfinite(p)

    o = o[mask_valid]
    p = p[mask_valid]

    if o.size == 0:
        return float("nan")

    threshold = np.quantile(o, q)
    mask = o >= threshold

    if not np.any(mask):
        return float("nan")

    return float(
        np.sqrt(np.mean((p[mask] - o[mask]) ** 2))
    )


def extreme_underprediction_frequency(obs, pred, event_id):
    """
    Fraction of events whose predicted peak is below the observed peak.

    Range:
        0 to 1

    Lower is better.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    return float(
        np.mean([x["underpredicted"] for x in events])
    )


def extreme_underprediction_magnitude(obs, pred, event_id):
    """
    Mean relative underprediction magnitude (%) across events
    that are underpredicted.

    Returns 0 if no events are underpredicted.
    """
    events = event_peak_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    vals = []

    for x in events:
        if x["underpredicted"]:
            vals.append(
                max(0.0, -x["signed_relative_error_pct"])
            )

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals))


def event_volume_errors(obs, pred, event_id, eps=1e-6):
    """
    Event-by-event volume error.

    Because dt is omitted, this compares sums over event windows.
    If a physical time step is needed, multiply both observed and
    predicted sums by dt before interpreting as volume.

    Returns
    -------
    list of dict
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    o, p, e = _clean_inputs(obs, pred, event_id)

    out = []

    for eid in np.unique(e):
        if eid < 0:
            continue

        mask = e == eid

        if not np.any(mask):
            continue

        obs_vol = float(np.sum(o[mask]))
        pred_vol = float(np.sum(p[mask]))

        error = pred_vol - obs_vol
        denom = max(abs(obs_vol), eps)

        out.append({
            "event": int(eid),
            "obs_volume": obs_vol,
            "pred_volume": pred_vol,
            "volume_error": float(error),
            "relative_volume_error_pct": float(
                100.0 * error / denom
            ),
            "absolute_relative_volume_error_pct": float(
                100.0 * abs(error) / denom
            ),
        })

    return out


def mean_absolute_event_volume_error(obs, pred, event_id):
    """
    Mean absolute relative event-volume error in percent.
    """
    events = event_volume_errors(obs, pred, event_id)

    if not events:
        return float("nan")

    return float(
        np.mean(
            [
                x["absolute_relative_volume_error_pct"]
                for x in events
            ]
        )
    )


def evaluate_peak_metrics(obs, pred, event_id):
    """
    Convenience wrapper for the main peak-oriented metrics.

    IMPORTANT:
    obs, pred, and event_id should come from a unique chronological
    test series, not flattened overlapping sequence windows.
    """
    return {
        "PeakRMSE": peak_rmse(obs, pred, event_id),
        "PeakMAE": peak_mae(obs, pred, event_id),
        "MeanRelativePeakError_percent":
            mean_relative_peak_error(obs, pred, event_id),
        "MedianRelativePeakError_percent":
            median_relative_peak_error(obs, pred, event_id),
        "RelativePeakBias_percent":
            relative_peak_bias(obs, pred, event_id),
        "Q95_RMSE": quantile_rmse(obs, pred, q=0.95),
        "Q99_RMSE": quantile_rmse(obs, pred, q=0.99),
        "ExtremeUnderpredictionFrequency":
            extreme_underprediction_frequency(obs, pred, event_id),
        "ExtremeUnderpredictionMagnitude_percent":
            extreme_underprediction_magnitude(obs, pred, event_id),
        "MeanAbsoluteEventVolumeError_percent":
            mean_absolute_event_volume_error(obs, pred, event_id),
    }
