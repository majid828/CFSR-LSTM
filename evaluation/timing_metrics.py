import numpy as np


_EPS = 1e-12


def _clean_inputs(obs, pred, event_id):
    """
    Convert inputs to aligned 1D arrays and remove positions
    where obs or pred is not finite.

    Parameters
    ----------
    obs : array-like
        Observed discharge.
    pred : array-like
        Predicted discharge.
    event_id : array-like
        Event label for each time step.
        Negative values are treated as non-event/background.

    Returns
    -------
    obs_clean, pred_clean, event_id_clean
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
        raise ValueError(
            "No finite paired observations remain after filtering."
        )

    return o, p, e


def event_timing_errors(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Compute event-by-event hard peak timing errors.

    For each event j:

        observed peak index:
            argmax_t Q_t

        predicted peak index:
            argmax_t Qhat_t

        signed timing error:
            (predicted_index - observed_index) * dt

    Interpretation
    --------------
    signed_timing_error > 0:
        predicted peak is late

    signed_timing_error < 0:
        predicted peak is early

    signed_timing_error = 0:
        exact peak timing

    Parameters
    ----------
    obs : array-like
        Observed discharge.
    pred : array-like
        Predicted discharge.
    event_id : array-like
        Event labels. Negative IDs are ignored.
    dt : float
        Time step size in desired physical units
        (e.g., hours or days).

    Returns
    -------
    list of dict
        One dictionary per valid event.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    o, p, e = _clean_inputs(obs, pred, event_id)

    out = []

    for eid in np.unique(e):
        if eid < 0:
            continue

        idx = np.where(e == eid)[0]

        if idx.size == 0:
            continue

        obs_event = o[idx]
        pred_event = p[idx]

        # local positions inside the event window
        local_obs_peak = int(np.argmax(obs_event))
        local_pred_peak = int(np.argmax(pred_event))

        # corresponding global indices in the reconstructed series
        obs_peak_index = int(idx[local_obs_peak])
        pred_peak_index = int(idx[local_pred_peak])

        signed_steps = pred_peak_index - obs_peak_index
        abs_steps = abs(signed_steps)

        event_length_steps = int(idx.size)

        signed_time = signed_steps * dt
        abs_time = abs_steps * dt

        event_duration = max(event_length_steps * dt, _EPS)

        relative_error = abs_time / event_duration

        out.append({
            "event": int(eid),

            "obs_peak_index": obs_peak_index,
            "pred_peak_index": pred_peak_index,

            "obs_peak_local_index": local_obs_peak,
            "pred_peak_local_index": local_pred_peak,

            "signed_timing_error_steps": int(signed_steps),
            "abs_timing_error_steps": int(abs_steps),

            "signed_timing_error": float(signed_time),
            "abs_timing_error": float(abs_time),

            "event_length_steps": event_length_steps,
            "event_duration": float(event_duration),

            "relative_timing_error": float(relative_error),
            "relative_timing_error_percent": float(
                100.0 * relative_error
            ),

            "early": bool(signed_steps < 0),
            "late": bool(signed_steps > 0),
            "exact": bool(signed_steps == 0),
        })

    return out


def mean_absolute_timing_error(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Mean absolute peak timing error across events.

    Lower is better.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean(
            [x["abs_timing_error"] for x in events]
        )
    )


def median_absolute_timing_error(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Median absolute peak timing error across events.

    More robust than the mean when a few events have
    very large timing errors.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.median(
            [x["abs_timing_error"] for x in events]
        )
    )


def timing_rmse(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Root mean squared peak timing error.

    Penalizes large timing errors more strongly.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    errors = np.asarray(
        [x["signed_timing_error"] for x in events],
        dtype=float,
    )

    return float(
        np.sqrt(np.mean(errors ** 2))
    )


def timing_bias(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Mean signed peak timing bias.

    Positive:
        model tends to predict peaks late.

    Negative:
        model tends to predict peaks early.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean(
            [x["signed_timing_error"] for x in events]
        )
    )


def mean_relative_timing_error(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Mean timing error normalized by event duration.

    Returns a fraction.

    Example:
        0.10 means the peak timing error is, on average,
        10% of the event duration.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean(
            [x["relative_timing_error"] for x in events]
        )
    )


def mean_relative_timing_error_percent(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Mean timing error as percent of event duration.
    """
    value = mean_relative_timing_error(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not np.isfinite(value):
        return float("nan")

    return float(100.0 * value)


def early_peak_fraction(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Fraction of events whose predicted peak occurs too early.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean([x["early"] for x in events])
    )


def late_peak_fraction(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Fraction of events whose predicted peak occurs too late.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean([x["late"] for x in events])
    )


def exact_peak_fraction(
    obs,
    pred,
    event_id,
    dt=1.0,
):
    """
    Fraction of events with exact peak timing.
    """
    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean([x["exact"] for x in events])
    )


def timing_within_tolerance(
    obs,
    pred,
    event_id,
    tolerance,
    dt=1.0,
):
    """
    Fraction of events whose absolute peak timing error
    is within a specified tolerance.

    Parameters
    ----------
    tolerance : float
        Maximum allowed absolute timing error in the
        same physical units as dt.

    Example
    -------
    If dt = 1 hour and tolerance = 3:

        result = fraction of events whose peak is predicted
        within +/- 3 hours.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")

    events = event_timing_errors(
        obs,
        pred,
        event_id,
        dt=dt,
    )

    if not events:
        return float("nan")

    return float(
        np.mean(
            [
                x["abs_timing_error"] <= tolerance
                for x in events
            ]
        )
    )


def evaluate_timing_metrics(
    obs,
    pred,
    event_id,
    dt=1.0,
    tolerances=None,
):
    """
    Convenience wrapper for peak timing evaluation.

    IMPORTANT
    ---------
    obs, pred, and event_id should represent a unique
    chronological time series.

    Do NOT pass flattened overlapping sequence windows.
    """
    if tolerances is None:
        tolerances = []

    result = {
        "MeanAbsTimingError":
            mean_absolute_timing_error(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "MedianAbsTimingError":
            median_absolute_timing_error(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "TimingRMSE":
            timing_rmse(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "TimingBias":
            timing_bias(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "MeanRelativeTimingError_percent":
            mean_relative_timing_error_percent(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "EarlyPeakFraction":
            early_peak_fraction(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "LatePeakFraction":
            late_peak_fraction(
                obs,
                pred,
                event_id,
                dt=dt,
            ),

        "ExactPeakFraction":
            exact_peak_fraction(
                obs,
                pred,
                event_id,
                dt=dt,
            ),
    }

    for tol in tolerances:
        result[
            f"TimingWithin_{tol:g}"
        ] = timing_within_tolerance(
            obs,
            pred,
            event_id,
            tolerance=tol,
            dt=dt,
        )

    return result
