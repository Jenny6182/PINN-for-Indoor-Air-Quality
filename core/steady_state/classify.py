"""
core/steady_state/classify.py
------------------------------
Classify timepoints within a segment as "steady" or "transient" based on
local gradient flatness, sustained over a minimum duration.

Used by the algebraic (non-PINN) steady-state pipeline: for scenario 1,
where a segment has both a transient and a steady portion, we isolate the
steady portion and solve for Q/S directly (no training).
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d


def classify_steady_transient(
    t_seg: np.ndarray,
    C_seg: np.ndarray,
    sigma: float = 1.5,
    eps_rel: float = 0.02,
    min_duration_frac: float = 0.15,
) -> np.ndarray:
    """
    Return a boolean mask over t_seg/C_seg marking steady-state points.

    Params
    ------
    sigma : smoothing applied to C before differentiating (same role as
            stage1's sigma -- keep consistent with your stage1 config
            unless you have a reason to decouple them).
    eps_rel : relative flatness threshold. A point is "flat" if
              |dC/dt| < eps_rel * (C_seg.max() - C_seg.min()).
              Relative, not absolute, so it doesn't need re-tuning across
              datasets with very different C magnitudes.
    min_duration_frac : minimum contiguous flat duration, as a fraction of
              the segment's own time span, required to call it "steady"
              rather than a momentary/noisy dip in the gradient.

    Returns
    -------
    steady_mask : bool array, same length as t_seg/C_seg.
    """
    t_seg = np.asarray(t_seg, dtype=np.float64)
    C_seg = np.asarray(C_seg, dtype=np.float64)
    n = len(t_seg)

    C_range = C_seg.max() - C_seg.min()
    if C_range < 1e-8:
        # segment is essentially flat throughout (no detectable transient) ->
        # treat everything as steady rather than dividing by ~0 below.
        return np.ones(n, dtype=bool)

    C_smooth = gaussian_filter1d(C_seg, sigma=sigma)
    dCdt = np.gradient(C_smooth, t_seg)

    threshold = eps_rel * C_range
    flat = np.abs(dCdt) < threshold

    min_duration = min_duration_frac * (t_seg[-1] - t_seg[0])
    steady_mask = np.zeros(n, dtype=bool)

    idx = 0
    while idx < n:
        if flat[idx]:
            j = idx
            while j < n and flat[j]:
                j += 1
            # contiguous flat run is [idx, j)
            if t_seg[j - 1] - t_seg[idx] >= min_duration:
                steady_mask[idx:j] = True
            idx = j
        else:
            idx += 1

    return steady_mask