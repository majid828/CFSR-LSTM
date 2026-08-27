import numpy as np


_EPS = 1e-12


def _as_prob_array(probs):
    """
    Convert routing probabilities to a float NumPy array.

    Expected shape:
        (..., L + 1)

    where the last axis contains probabilities for lags
        0, 1, ..., L.
    """
    p = np.asarray(probs, dtype=float)

    if p.ndim < 1:
        raise ValueError("probs must have at least one dimension.")

    if p.shape[-1] < 1:
        raise ValueError("The lag dimension must contain at least one value.")

    return p


def validate_routing_probs(
    probs,
    atol=1e-6,
    renormalize=False,
):
    """
    Validate routing probabilities.

    Checks:
    - finite values
    - nonnegative values
    - probabilities sum to 1 along the lag axis

    Parameters
    ----------
    probs : array-like
        Routing probabilities with shape (..., L+1).
    atol : float
        Tolerance for checking probability sums.
    renormalize : bool
        If True, negative numerical noise is clipped to zero and
        probabilities are renormalized along the lag axis.

    Returns
    -------
    np.ndarray
        Validated routing probabilities.
    """
    p = _as_prob_array(probs)

    if not np.all(np.isfinite(p)):
        raise ValueError("Routing probabilities contain NaN or inf values.")

    if renormalize:
        p = np.maximum(p, 0.0)

        denom = np.sum(p, axis=-1, keepdims=True)

        if np.any(denom <= _EPS):
            raise ValueError(
                "At least one routing distribution has zero total mass."
            )

        p = p / denom
        return p

    if np.any(p < -atol):
        raise ValueError("Routing probabilities contain negative values.")

    sums = np.sum(p, axis=-1)

    if not np.allclose(sums, 1.0, atol=atol):
        raise ValueError(
            "Routing probabilities do not sum to 1 along the lag axis."
        )

    return p


def expected_delay_from_probs(
    probs,
    dt=1.0,
    validate=True,
):
    """
    Compute expected routing delay.

    For routing probabilities pi_{t,l}:

        tau_t = sum_l l * dt * pi_{t,l}

    Parameters
    ----------
    probs : array-like
        Routing probabilities with shape (..., L+1).
    dt : float
        Time step size.
    validate : bool
        Whether to validate that probabilities are proper distributions.

    Returns
    -------
    np.ndarray
        Expected delay for each routing distribution.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    p = (
        validate_routing_probs(probs)
        if validate
        else _as_prob_array(probs)
    )

    lags = np.arange(p.shape[-1], dtype=float)

    return np.sum(
        p * lags,
        axis=-1
    ) * dt


def modal_lag_from_probs(
    probs,
    dt=1.0,
    validate=True,
):
    """
    Return the most probable routing lag.

    This is different from expected delay.

    Returns
    -------
    np.ndarray
        Modal routing lag expressed in physical time units.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    p = (
        validate_routing_probs(probs)
        if validate
        else _as_prob_array(probs)
    )

    return np.argmax(p, axis=-1).astype(float) * dt


def routing_entropy(
    probs,
    eps=1e-12,
    validate=True,
):
    """
    Shannon entropy of the routing distribution.

        H_t = -sum_l pi_{t,l} log(pi_{t,l})

    Interpretation
    --------------
    Low entropy:
        routing concentrated at a small number of lags.

    High entropy:
        routing spread across many lags.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    p = (
        validate_routing_probs(probs)
        if validate
        else _as_prob_array(probs)
    )

    return -np.sum(
        p * np.log(p + eps),
        axis=-1
    )


def normalized_routing_entropy(
    probs,
    eps=1e-12,
    validate=True,
):
    """
    Routing entropy normalized to approximately [0, 1].

    0:
        maximally concentrated routing distribution.

    1:
        approximately uniform distribution across all lags.
    """
    p = (
        validate_routing_probs(probs)
        if validate
        else _as_prob_array(probs)
    )

    H = routing_entropy(
        p,
        eps=eps,
        validate=False,
    )

    n_lags = p.shape[-1]

    if n_lags <= 1:
        return np.zeros_like(H)

    return H / np.log(n_lags)


def routing_concentration(
    probs,
    validate=True,
):
    """
    Maximum routing probability at each generation time.

    High values:
        strongly concentrated routing.

    Low values:
        more diffuse routing.
    """
    p = (
        validate_routing_probs(probs)
        if validate
        else _as_prob_array(probs)
    )

    return np.max(p, axis=-1)


def effective_number_of_lags(
    probs,
    eps=1e-12,
    validate=True,
):
    """
    Effective number of active routing lags.

        N_eff = exp(H)

    where H is Shannon entropy.

    Interpretation:
        close to 1  -> routing concentrated at one lag
        larger      -> routing spread over several lags
    """
    H = routing_entropy(
        probs,
        eps=eps,
        validate=validate,
    )

    return np.exp(H)


def delay_summary(
    probs,
    dt=1.0,
    validate=True,
):
    """
    Summarize learned expected routing delays.

    Returns
    -------
    dict
        Summary statistics of expected routing delay.
    """
    d = expected_delay_from_probs(
        probs,
        dt=dt,
        validate=validate,
    )

    d = np.asarray(d, dtype=float).reshape(-1)

    finite = np.isfinite(d)
    d = d[finite]

    if d.size == 0:
        return {
            "mean_delay": np.nan,
            "median_delay": np.nan,
            "std_delay": np.nan,
            "min_delay": np.nan,
            "max_delay": np.nan,
            "q25_delay": np.nan,
            "q75_delay": np.nan,
        }

    return {
        "mean_delay": float(np.mean(d)),
        "median_delay": float(np.median(d)),
        "std_delay": float(np.std(d)),
        "min_delay": float(np.min(d)),
        "max_delay": float(np.max(d)),
        "q25_delay": float(np.quantile(d, 0.25)),
        "q75_delay": float(np.quantile(d, 0.75)),
    }


def routing_summary(
    probs,
    dt=1.0,
):
    """
    Comprehensive routing summary.

    Returns
    -------
    dict
        Mean/median delay, entropy, concentration,
        modal lag, and effective lag support.
    """
    p = validate_routing_probs(probs)

    delay = expected_delay_from_probs(
        p,
        dt=dt,
        validate=False,
    ).reshape(-1)

    mode = modal_lag_from_probs(
        p,
        dt=dt,
        validate=False,
    ).reshape(-1)

    entropy = routing_entropy(
        p,
        validate=False,
    ).reshape(-1)

    norm_entropy = normalized_routing_entropy(
        p,
        validate=False,
    ).reshape(-1)

    concentration = routing_concentration(
        p,
        validate=False,
    ).reshape(-1)

    effective_lags = effective_number_of_lags(
        p,
        validate=False,
    ).reshape(-1)

    return {
        "mean_expected_delay": float(np.mean(delay)),
        "median_expected_delay": float(np.median(delay)),
        "std_expected_delay": float(np.std(delay)),
        "mean_modal_lag": float(np.mean(mode)),
        "median_modal_lag": float(np.median(mode)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_normalized_entropy": float(np.mean(norm_entropy)),
        "mean_concentration": float(np.mean(concentration)),
        "mean_effective_lags": float(np.mean(effective_lags)),
    }


def routing_lag_distribution(
    probs,
    validate=True,
):
    """
    Mean routing probability assigned to each lag.

    Returns
    -------
    np.ndarray
        Mean probability for lag 0...L.
    """
    p = (
        validate_routing_probs(probs)
        if validate
        else _as_prob_array(probs)
    )

    axes = tuple(range(p.ndim - 1))

    return np.mean(p, axis=axes)


def delay_correlation(
    probs,
    variable,
    dt=1.0,
):
    """
    Pearson correlation between expected routing delay and
    an external hydrologic variable.

    Example variables:
    - precipitation intensity
    - antecedent wetness
    - observed discharge
    - fast-response magnitude

    Parameters
    ----------
    probs : array-like
        Routing probabilities, shape (..., L+1).
    variable : array-like
        Hydrologic variable aligned with the routing generation times.
    dt : float
        Time step.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    delay = expected_delay_from_probs(
        probs,
        dt=dt,
    ).reshape(-1)

    x = np.asarray(variable, dtype=float).reshape(-1)

    if delay.shape != x.shape:
        raise ValueError(
            f"Expected delay and variable must have the same length, "
            f"got {delay.shape} and {x.shape}"
        )

    mask = np.isfinite(delay) & np.isfinite(x)

    delay = delay[mask]
    x = x[mask]

    if delay.size < 2:
        return float("nan")

    if np.std(delay) <= _EPS or np.std(x) <= _EPS:
        return float("nan")

    return float(
        np.corrcoef(delay, x)[0, 1]
    )
