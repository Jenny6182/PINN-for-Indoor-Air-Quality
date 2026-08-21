"""
Stage 1 replacement: label each timestep as 'steady' or 'transient'
using a smoothed derivative of the measured concentration, then cut
segments at transient-after-steady boundaries.
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

    A point is a candidate steady point if |dC/dt| falls below the
    'slope_percentile' of the whole trace's |dC/dt| distribution.
    Candidates are only kept as steady if they're part of a run of at
    least 'min_run' consecutive candidate points.
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
