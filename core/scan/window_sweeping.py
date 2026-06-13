"""
stage1.py
---------
Stage I of RAA-PINN: sliding window least squares scan.

The ODE is V*dC/dt = Q*(C_out - C) + S.
On each window of size window_size, we fit Q and S by least squares.
Windows straddling a true changepoint produce high residuals because
no single constant Q, S can satisfy the ODE across both regimes.
The residual score at each timepoint is the detection signal.

Functions:
    - stage1_scan() - run the sliding window scan, return scores
    - find_candidate_intervals() - find peaks and return intervals around them
    - find_top_k_intervals() - find intervals of top k peaks, for example, if asked to find top 5 peaks, 
                               it returns 5 interval given that it wouldn't disobey the minimum space between windows
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def stage1_scan(t: np.ndarray, C_meas: np.ndarray, V: float, C_out: float, window_size: int=20, sigma: float=1.5):
    """
    Slide a window across the time series, fit Q and S by least squares
    on each window, and record the ODE residual as the score

    Special params
    window_size: int   — number of points per window (must be > 2)
    sigma: float — Gaussian smoothing sigma applied before differentiation.
                Higher sigma = smoother derivative but 
                Reduce to 0.5–1.0 if peaks are weak

    Return
    scores : np.ndarray, shape (N,), how inconsistent at each timepoint. 
             NaN at the first and last window_size//2 points
             because a full window cannot be centered there
    """
    t = t.flatten().astype(np.float64) # flatten to 1D
    C_meas = C_meas.flatten().astype(np.float64) # flatten to 1D

    C_smooth = gaussian_filter1d(C_meas, sigma=sigma) # smooth C with gaussian filter before differentiation 
    dC_dt = np.gradient(C_smooth, t) # estimate dC/dt via np.gradient on the smoothed signal

    n = len(t)
    half = window_size // 2 # starting center point of first window
    scores = np.full(n, np.nan) # create output array of length of t, initialized to nan

    # Calculate score at each window centered around i
    for i in range(half, n - half): # from the first center point to last center point
        C_win = C_smooth[i-half : i+half] # window of C
        dCdt_win = dC_dt[i-half : i+half]

        # Q(C_out - C) + S = V * dC/dt, rewrite as Ax = b so x = [Q, S] and A = [C_out - C, 1]
        A = np.column_stack([C_out - C_win, np.ones(window_size)])
        b = V * dCdt_win

        # least squares — finds best single Q and S for this window
        params, _, _, _ = lstsq(A, b, rcond=None) # use scipy least square method to solve for best possible Q, S

        # score = mean squared ODE violation after best fit
        scores[i] = np.mean((A @ params - b) ** 2)

    return scores


def find_candidate_intervals(t, scores, prominence, distance, margin_h=0.3):
    """
    Find peaks in the score array and return a candidate interval around each.

    Params
    ----------
    prominence: float, min prominence a peak must have to be detected (higher, less false positive, but get less small jump)
    distance: int, minimum number of timepoints between two peaks (should be roughly window_size)
              to prevent detecting same point twice or too close to each other
    margin_h: float, half-width of candidate interval around each peak time where we'll search for the changepoint

    Returns
    -------
    peak_indices: np.ndarray, indices into t of detected peaks
    intervals: list of (t_left, t_right) tuples, one per peak
    """
    t_flat = t.flatten()

    # replace nan with 0 so find_peaks doesn't skip edge regions
    scores_clean = np.nan_to_num(scores, nan=0.0)

    peak_indices, _ = find_peaks(scores_clean,
                                 prominence=prominence,
                                 distance=distance) # use scipy find_peaks

    intervals = []
    for idx in peak_indices:
        t_peak  = t_flat[idx]
        t_left  = max(t_flat[0],  t_peak - margin_h)
        t_right = min(t_flat[-1], t_peak + margin_h)
        intervals.append((t_left, t_right))

    return peak_indices, intervals



def find_top_k_intervals(t, scores, k, margin_h=0.3, min_distance_h=0.4):
    """
    Return the top-k candidate intervals by picking the k highest score peaks,
    enforcing a minimum separation between selected peaks.

    Params
    ----------
    k: int, exact number of changepoints expected, will search for this many if doesn't disobey min_distance
    min_distance_h: float, minimum time gap between two selected peaks [hours].
                             Prevents two peaks from the same changepoint
                             being double-counted. Set to roughly your window
                             duration in hours

    Returns
    -------
    peak_indices: np.ndarray, indices of the k selected peaks, sorted by time
    intervals: list of (t_left, t_right) tuples, one per peak, sorted by time
    """
    t_flat = t.flatten()
    scores_clean = np.nan_to_num(scores, nan=0.0)

    # sort all timepoints by score descending, pick top-k while respecting min separation
    order = np.argsort(scores_clean)[::-1] # sort by descending order

    selected = []
    for idx in order:
        if len(selected) == k: # if we found k top peaks break
            break
        if any(abs(t_flat[idx] - t_flat[s]) < min_distance_h for s in selected): 
            # reject if too close to an already-selected peak
            continue
        selected.append(idx)

    # sort selected peaks ascending
    selected = sorted(selected, key=lambda i: t_flat[i])
    peak_indices = np.array(selected)

    intervals = []
    for idx in peak_indices:
        t_peak = t_flat[idx]
        t_left = max(t_flat[0],  t_peak - margin_h)
        t_right = min(t_flat[-1], t_peak + margin_h)
        intervals.append((t_left, t_right))

    return peak_indices, intervals