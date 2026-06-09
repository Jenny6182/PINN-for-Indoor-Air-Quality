"""
stage1.py
---------
Stage I of RAA-PINN: sliding window least squares scan.

The ODE is V*dC/dt = Q*(C_out - C) + S.
On each window of size window_size, we fit Q and S by least squares.
Windows straddling a true changepoint produce high residuals because
no single constant Q, S can satisfy the ODE across both regimes.
The residual score at each timepoint is the detection signal.

Main functions:
    stage1_scan()              — run the sliding window scan, return scores
    find_candidate_intervals() — find peaks and return intervals around them
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def stage1_scan(t, C_meas, V, C_out, window_size=20, sigma=1.5):
    """
    Slide a window across the time series, fit Q and S by least squares
    on each window, and record the ODE residual as the score.

    Parameters
    ----------
    t          : np.ndarray, shape (N,) or (N,1) — time in hours
    C_meas     : np.ndarray, shape (N,) or (N,1) — noisy CO2 measurements
    V          : float — room volume [m^3]
    C_out      : float — outdoor CO2 [ppm]
    window_size: int   — number of points per window (must be > 2)
    sigma      : float — Gaussian smoothing sigma applied before differentiation.
                         Higher sigma = smoother derivative but may blur peaks.
                         Reduce to 0.5–1.0 if peaks are weak.

    Returns
    -------
    scores : np.ndarray, shape (N,) — physics inconsistency score at each
             timepoint. NaN at the first and last window_size//2 points
             because a full window cannot be centred there.

    How it works
    ------------
    At each window centred on timepoint i:
      1. Smooth C_meas with a Gaussian to reduce noise before differentiation
      2. Estimate dC/dt numerically using np.gradient on the smoothed signal
      3. Build the linear system:
             [C_out - C[j],  1] @ [Q, S]  =  V * dC_dt[j]
         for every point j in the window — this is the ODE rearranged
      4. Solve for the best single Q and S by least squares
      5. Compute the mean squared ODE violation after that best fit
         — this is the score for timepoint i

    Scores are high where no single (Q, S) can satisfy the ODE on the window,
    which happens when Q or S jumps somewhere inside the window.
    """
    # flatten to 1D
    t      = t.flatten().astype(np.float64)
    C_meas = C_meas.flatten().astype(np.float64)

    # smooth before differentiation — raw noisy derivative is unusable
    C_smooth = gaussian_filter1d(C_meas, sigma=sigma)
    dC_dt    = np.gradient(C_smooth, t)

    n      = len(t)
    half   = window_size // 2
    scores = np.full(n, np.nan)

    for i in range(half, n - half):
        idx = slice(i - half, i + half)

        C_win    = C_smooth[idx]
        dCdt_win = dC_dt[idx]

        # linear system: [C_out - C, 1] @ [Q, S] = V * dC/dt
        # two columns: first is the Q feature, second is the S feature (constant 1)
        A = np.column_stack([C_out - C_win, np.ones(window_size)])
        b = V * dCdt_win

        # least squares — finds best single Q and S for this window
        params, _, _, _ = lstsq(A, b, rcond=None)

        # score = mean squared ODE violation after best fit
        scores[i] = np.mean((A @ params - b) ** 2)

    return scores


def find_candidate_intervals(t, scores, prominence, distance, margin_h=0.3):
    """
    Find peaks in the score array and return a candidate interval around each.

    Parameters
    ----------
    t          : np.ndarray, shape (N,) or (N,1) — time in hours
    scores     : np.ndarray, shape (N,) — output of stage1_scan
    prominence : float — minimum prominence a peak must have to be detected.
                         Higher = fewer false positives but may miss small jumps.
                         Start with 10–20% of the largest score value.
    distance   : int   — minimum number of timepoints between two peaks.
                         Prevents detecting the same changepoint twice.
                         Set to roughly window_size.
    margin_h   : float — half-width of candidate interval around each peak [hours].
                         Stage II will search for the exact changepoint within
                         [peak_time - margin_h, peak_time + margin_h].

    Returns
    -------
    peak_indices : np.ndarray — indices into t of detected peaks
    intervals    : list of (t_left, t_right) tuples, one per peak
    """
    t_flat = t.flatten()

    # replace NaN with 0 so find_peaks doesn't skip edge regions
    scores_clean = np.nan_to_num(scores, nan=0.0)

    peak_indices, _ = find_peaks(scores_clean,
                                 prominence=prominence,
                                 distance=distance)

    intervals = []
    for idx in peak_indices:
        t_peak  = t_flat[idx]
        t_left  = max(t_flat[0],  t_peak - margin_h)
        t_right = min(t_flat[-1], t_peak + margin_h)
        intervals.append((t_left, t_right))

    return peak_indices, intervals