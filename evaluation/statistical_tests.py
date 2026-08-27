import numpy as np
from scipy.stats import wilcoxon, ttest_rel


_EPS = 1e-12


def _clean_paired(a, b):
    """
    Convert two paired samples to aligned 1D float arrays
    and remove pairs containing NaN or inf.

    Parameters
    ----------
    a, b : array-like
        Paired observations.

    Returns
    -------
    a_clean, b_clean : np.ndarray
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have the same shape, got {a.shape} and {b.shape}"
        )

    mask = np.isfinite(a) & np.isfinite(b)

    a = a[mask]
    b = b[mask]

    return a, b


def paired_mean_difference(a, b):
    """
    Mean paired difference:

        mean(a - b)

    Interpretation depends on how a and b are defined.

    Example for an error metric:
        a = Standard LSTM error
        b = CFSR-LSTM error

    Positive mean difference means CFSR-LSTM has lower error.
    """
    a, b = _clean_paired(a, b)

    if len(a) == 0:
        return float("nan")

    return float(np.mean(a - b))


def paired_median_difference(a, b):
    """
    Median paired difference:

        median(a - b)
    """
    a, b = _clean_paired(a, b)

    if len(a) == 0:
        return float("nan")

    return float(np.median(a - b))


def paired_cohens_d(a, b):
    """
    Paired-sample Cohen's d.

        d = mean(a - b) / std(a - b)

    Uses sample standard deviation (ddof=1).

    Returns NaN if fewer than 2 paired samples exist or
    if the difference variance is essentially zero.
    """
    a, b = _clean_paired(a, b)

    if len(a) < 2:
        return float("nan")

    d = a - b
    sd = np.std(d, ddof=1)

    if sd <= _EPS:
        return float("nan")

    return float(np.mean(d) / sd)


def paired_tests(a, b, alternative="two-sided"):
    """
    Perform paired statistical tests.

    Tests
    -----
    1. Wilcoxon signed-rank test
    2. Paired t-test

    Parameters
    ----------
    a, b : array-like
        Paired values.

        For lower-is-better error metrics, a useful convention is:

            a = Standard LSTM error
            b = CFSR-LSTM error

        Then a positive difference (a - b) indicates improvement.

    alternative : {"two-sided", "greater", "less"}
        Statistical alternative hypothesis.

        For scipy tests:
            "greater" tests whether a tends to be greater than b.
            "less" tests whether a tends to be less than b.

    Returns
    -------
    dict
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'."
        )

    a, b = _clean_paired(a, b)

    n = len(a)

    result = {
        "n": int(n),
        "mean_a": float(np.mean(a)) if n else np.nan,
        "mean_b": float(np.mean(b)) if n else np.nan,
        "mean_difference_a_minus_b": (
            float(np.mean(a - b)) if n else np.nan
        ),
        "median_difference_a_minus_b": (
            float(np.median(a - b)) if n else np.nan
        ),
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p": np.nan,
        "paired_t_statistic": np.nan,
        "paired_t_p": np.nan,
        "cohens_d_paired": np.nan,
    }

    if n < 2:
        return result

    diff = a - b

    # --------------------------------------------------
    # Paired effect size
    # --------------------------------------------------
    result["cohens_d_paired"] = paired_cohens_d(a, b)

    # --------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------
    #
    # If every paired difference is zero, the test is
    # degenerate. In that case p=1 is appropriate because
    # there is no evidence of a difference.
    #
    if np.all(np.abs(diff) <= _EPS):
        result["wilcoxon_statistic"] = 0.0
        result["wilcoxon_p"] = 1.0

    else:
        try:
            w = wilcoxon(
                a,
                b,
                alternative=alternative,
                zero_method="wilcox",
            )

            result["wilcoxon_statistic"] = float(w.statistic)
            result["wilcoxon_p"] = float(w.pvalue)

        except ValueError:
            result["wilcoxon_statistic"] = np.nan
            result["wilcoxon_p"] = np.nan

    # --------------------------------------------------
    # Paired t-test
    # --------------------------------------------------
    #
    # If differences have exactly zero variance:
    # - all differences zero -> p = 1
    # - constant nonzero difference -> mathematically
    #   degenerate for a t test, so report NaN rather than
    #   pretending the result is well-defined.
    #
    sd_diff = np.std(diff, ddof=1)

    if sd_diff <= _EPS:

        if abs(np.mean(diff)) <= _EPS:
            result["paired_t_statistic"] = 0.0
            result["paired_t_p"] = 1.0
        else:
            result["paired_t_statistic"] = np.nan
            result["paired_t_p"] = np.nan

    else:
        t = ttest_rel(
            a,
            b,
            alternative=alternative,
        )

        result["paired_t_statistic"] = float(t.statistic)
        result["paired_t_p"] = float(t.pvalue)

    return result


def bootstrap_mean_difference(
    a,
    b,
    n_boot=10000,
    seed=42,
    confidence=0.95,
):
    """
    Paired bootstrap confidence interval for mean(a - b).

    Pairing is preserved by bootstrapping the difference vector.

    Parameters
    ----------
    a, b : array-like
        Paired values.
    n_boot : int
        Number of bootstrap samples.
    seed : int
        Random seed.
    confidence : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    dict
    """
    if n_boot < 100:
        raise ValueError("n_boot should be at least 100.")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    a, b = _clean_paired(a, b)

    n = len(a)

    if n == 0:
        return {
            "n": 0,
            "mean_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "confidence": confidence,
        }

    d = a - b

    rng = np.random.default_rng(seed)

    # More memory-efficient than repeatedly appending Python lists.
    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means[i] = np.mean(d[indices])

    alpha = 1.0 - confidence

    lo, hi = np.quantile(
        boot_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return {
        "n": int(n),
        "mean_difference": float(np.mean(d)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": float(confidence),
    }


def bootstrap_median_difference(
    a,
    b,
    n_boot=10000,
    seed=42,
    confidence=0.95,
):
    """
    Paired bootstrap confidence interval for median(a - b).

    Useful when event-level errors are skewed or contain outliers.
    """
    if n_boot < 100:
        raise ValueError("n_boot should be at least 100.")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    a, b = _clean_paired(a, b)

    n = len(a)

    if n == 0:
        return {
            "n": 0,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "confidence": confidence,
        }

    d = a - b

    rng = np.random.default_rng(seed)

    boot_medians = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot_medians[i] = np.median(d[indices])

    alpha = 1.0 - confidence

    lo, hi = np.quantile(
        boot_medians,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return {
        "n": int(n),
        "median_difference": float(np.median(d)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": float(confidence),
    }


def win_rate(a, b, lower_is_better=True):
    """
    Fraction of paired cases where model/value b is better than a.

    Recommended convention
    ----------------------
    a = Standard LSTM
    b = CFSR-LSTM

    For error metrics:
        lower_is_better=True

    For efficiency metrics such as NSE/KGE:
        lower_is_better=False

    Returns
    -------
    dict
    """
    a, b = _clean_paired(a, b)

    n = len(a)

    if n == 0:
        return {
            "n": 0,
            "b_win_rate": np.nan,
            "a_win_rate": np.nan,
            "tie_rate": np.nan,
        }

    if lower_is_better:
        b_wins = b < a
        a_wins = a < b
    else:
        b_wins = b > a
        a_wins = a > b

    ties = np.isclose(a, b)

    return {
        "n": int(n),
        "b_win_rate": float(np.mean(b_wins)),
        "a_win_rate": float(np.mean(a_wins)),
        "tie_rate": float(np.mean(ties)),
    }


def compare_models(
    baseline,
    proposed,
    lower_is_better=True,
    n_boot=10000,
    seed=42,
):
    """
    Convenience function for Standard LSTM vs CFSR-LSTM.

    Parameters
    ----------
    baseline : array-like
        Paired metric values for Standard LSTM.
    proposed : array-like
        Paired metric values for CFSR-LSTM.
    lower_is_better : bool
        True for metrics such as RMSE, peak error, timing error.
        False for metrics such as NSE or KGE.

    Interpretation
    --------------
    baseline - proposed > 0
        indicates improvement when lower is better.

    proposed - baseline > 0
        indicates improvement when higher is better.
    """
    baseline, proposed = _clean_paired(
        baseline,
        proposed,
    )

    if lower_is_better:
        alternative = "greater"
        a = baseline
        b = proposed
    else:
        # Reorder so positive a-b still means CFSR improvement.
        alternative = "greater"
        a = proposed
        b = baseline

    tests = paired_tests(
        a,
        b,
        alternative=alternative,
    )

    bootstrap = bootstrap_mean_difference(
        a,
        b,
        n_boot=n_boot,
        seed=seed,
    )

    wins = win_rate(
        baseline,
        proposed,
        lower_is_better=lower_is_better,
    )

    return {
        "n": tests["n"],
        "mean_improvement": tests["mean_difference_a_minus_b"],
        "median_improvement": tests["median_difference_a_minus_b"],
        "wilcoxon_statistic": tests["wilcoxon_statistic"],
        "wilcoxon_p": tests["wilcoxon_p"],
        "paired_t_statistic": tests["paired_t_statistic"],
        "paired_t_p": tests["paired_t_p"],
        "cohens_d_paired": tests["cohens_d_paired"],
        "bootstrap_ci95_low": bootstrap["ci_low"],
        "bootstrap_ci95_high": bootstrap["ci_high"],
        "proposed_win_rate": wins["b_win_rate"],
        "baseline_win_rate": wins["a_win_rate"],
        "tie_rate": wins["tie_rate"],
    }
