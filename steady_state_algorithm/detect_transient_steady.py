"""
Stage 1 replacement: label each timestep as 'steady' or 'transient'
using a smoothed derivative of the measured concentration, then cut
segments at transient-after-steady boundaries.

No physics loss, no PINN — just a rolling-slope threshold.
"""
import numpy as np
from scipy.signal import savgol_filter


def smoothed_derivative(t, C, window_length=7, polyorder=2):
    """
    Savitzky-Golay smoothed dC/dt. Robust to measurement noise
    (C_meas vs C_true) without lagging like a moving average would.
    window_length must be odd and <= len(C).
    """
    n = len(C)
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < polyorder + 2:
        # too few points to smooth meaningfully; fall back to raw diff
        dCdt = np.gradient(C, t)
        return dCdt
    dt = np.median(np.diff(t))
    dCdt = savgol_filter(C, window_length=wl, polyorder=polyorder,
                          deriv=1, delta=dt)
    return dCdt


def label_steady(t, C, window_length=7, polyorder=2,
                  slope_percentile=25, min_run=5):
    """
    Returns a boolean array: True = steady, False = transient.

    A point is a *candidate* steady point if |dC/dt| falls below the
    `slope_percentile` of the whole trace's |dC/dt| distribution.
    Candidates are only kept as steady if they're part of a run of at
    least `min_run` consecutive candidate points.
    """
    dCdt = smoothed_derivative(t, C, window_length, polyorder)
    abs_slope = np.abs(dCdt)
    thresh = np.percentile(abs_slope, slope_percentile)
    candidate = abs_slope < thresh

    steady = np.zeros_like(candidate, dtype=bool)
    run_start = None
    for i, is_cand in enumerate(candidate):
        if is_cand and run_start is None:
            run_start = i
        elif not is_cand and run_start is not None:
            if i - run_start >= min_run:
                steady[run_start:i] = True
            run_start = None
    if run_start is not None and len(candidate) - run_start >= min_run:
        steady[run_start:] = True

    return steady

def label_steady_threshold(t, C, window_length=7, polyorder=2,
                           slope_threshold=0.15, min_run=5):
    """
    Returns a boolean array: True = steady, False = transient.

    A point is considered steady if the smoothed derivative magnitude
    is below a fixed threshold determined from expected noise-induced
    slope variation.
    """
    dCdt = smoothed_derivative(t, C, window_length, polyorder)
    abs_slope = np.abs(dCdt)

    candidate = abs_slope < slope_threshold

    steady = np.zeros_like(candidate, dtype=bool)
    run_start = None

    for i, is_cand in enumerate(candidate):
        if is_cand and run_start is None:
            run_start = i
        elif not is_cand and run_start is not None:
            if i - run_start >= min_run:
                steady[run_start:i] = True
            run_start = None

    if run_start is not None and len(candidate) - run_start >= min_run:
        steady[run_start:] = True

    return steady


def local_slope_significance(t, C, window, sigma_meas):
    """
    For each point, fit a local linear regression of C vs t over a
    centered window of `window` points, and return the t-statistic of
    the fitted slope: slope / standard_error(slope).

    Large |t_stat| -> the local trend is unlikely to be noise alone.
    Small |t_stat| -> consistent with pure noise, regardless of how
    large the raw slope looks -- this is what lets a short, high-slope
    noise spike and a long, shallow real trend get told apart, instead
    of both being judged against the same fixed slope cutoff.

    Standard error is computed directly from the known sigma_meas
    (not estimated from residuals), since the noise level is assumed
    known from the sensor/generator rather than fit from data.

    Returns an array the same length as t/C, with NaN at the edges
    (where a full window isn't available).
    """
    n = len(t)
    half = window // 2
    t_stat = np.full(n, np.nan)

    for i in range(half, n - half):
        tt = t[i - half:i + half + 1]
        CC = C[i - half:i + half + 1]

        tt_c = tt - tt.mean()
        Sxx = np.sum(tt_c ** 2)

        slope = np.sum(tt_c * (CC - CC.mean())) / Sxx
        se_slope = sigma_meas / np.sqrt(Sxx)

        t_stat[i] = slope / se_slope

    return t_stat


# def label_steady_significance(t, C, window=21, sigma_meas=15.0, crit=3.0, min_run=5):
#     """
#     Alternative to label_steady: classify points as steady/transient
#     using the significance of a local linear trend rather than a fixed
#     or percentile-based threshold on the raw smoothed derivative.

#     Returns a boolean array: True = steady, False = transient.

#     A point is a *candidate* steady point if its local slope t-statistic
#     (see local_slope_significance) falls below `crit` in magnitude --
#     i.e. the local trend is not statistically distinguishable from noise
#     given sigma_meas. Candidates are only kept as steady if part of a
#     run of at least `min_run` consecutive candidate points, same
#     run-length logic as label_steady.

#     window:     number of points in the local regression window
#                 (analogous role to label_steady's window_length, but
#                 tune independently -- this one trades detection
#                 latency near changepoints for statistical power)
#     sigma_meas: known measurement noise std on C
#     crit:       significance threshold on |t_stat|; crit=2.0 is
#                 roughly a 95% two-sided cutoff, crit=3.0 roughly 99.7%
#     min_run:    minimum consecutive candidate points to confirm a
#                 steady run (same purpose as in label_steady)
#     """
#     t_stat = local_slope_significance(t, C, window, sigma_meas)
#     crit = 1.5
#     raw_transient = np.abs(t_stat) >= crit
#     print("raw transient point count:", raw_transient.sum())
#     print("raw transient fraction:", raw_transient.mean())

#     print("Type of crit:", type(crit))
#     print("Type of t_stat:", type(t_stat))

#     candidate = np.abs(t_stat) < crit
#     candidate = np.nan_to_num(candidate, nan=False).astype(bool)

#     steady = np.zeros_like(candidate, dtype=bool)
#     run_start = None
#     for i, is_cand in enumerate(candidate):
#         if is_cand and run_start is None:
#             run_start = i
#         elif not is_cand and run_start is not None:
#             if i - run_start >= min_run:
#                 steady[run_start:i] = True
#             run_start = None
#     if run_start is not None and len(candidate) - run_start >= min_run:
#         steady[run_start:] = True

#     return steady


def _filter_short_runs(mask, min_len, fill_value):
    """Flip any run of True in `mask` shorter than `min_len` to `fill_value`."""
    mask = mask.copy()
    n = len(mask)
    run_start = None
    for i in range(n):
        if mask[i] and run_start is None:
            run_start = i
        elif not mask[i] and run_start is not None:
            if i - run_start < min_len:
                mask[run_start:i] = fill_value
            run_start = None
    if run_start is not None and n - run_start < min_len:
        mask[run_start:] = fill_value
    return mask


def label_steady_significance_v2(t, C, window=21, sigma_meas=15.0, crit=1.5,
                                  min_steady_run=5, min_transient_run=8, debug=False):
    """
    Overdetect transients (loose crit), then discard short transient
    runs as noise before confirming steady runs. See _filter_short_runs.
    """
    t_stat = local_slope_significance(t, C, window, sigma_meas)
    is_transient = np.nan_to_num(np.abs(t_stat) >= crit, nan=True).astype(bool)

    if debug:
        print("raw transient point count:", is_transient.sum())
        print("raw transient fraction:", is_transient.mean())

    is_transient = _filter_short_runs(is_transient, min_transient_run, fill_value=False)
    steady_candidate = ~is_transient
    steady = _filter_short_runs(steady_candidate, min_steady_run, fill_value=False)

    return steady