"""
evaluation.py
-------------

Responsibility: evaluate predictions against ground truth

Metrics for comparing predicted (estimated) piecewise-constant parameters
against ground truth, without requiring explicit segment-to-segment
correspondence.

Two families:
    - reconstruction_error: dense pointwise comparison of true vs predicted
      step function, robust to wrong number/position of changepoints.
    - changepoint_metrics: precision/recall/F1 on changepoint locations
      within a tolerance window, plus segment count error.
"""

from __future__ import annotations

import numpy as np


def _build_step_func(taus, values, t_min, t_max):
    """
    Build a callable step function from changepoints + segment values.

    taus:   sorted list/array of interior changepoints (len = n_segments - 1)
    values: segment values (len = n_segments)
    """
    taus = np.asarray(taus, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(values) != len(taus) + 1:
        raise ValueError(
            f"values must have len(taus)+1 entries, got {len(values)} values "
            f"and {len(taus)} taus"
        )

    boundaries = np.concatenate(([t_min], np.sort(taus), [t_max]))

    def f(t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(boundaries, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return f


def reconstruction_error(t_grid, param_true, pred_taus, pred_values, t_min, t_max):
    """
    Dense pointwise comparison of the true param(t) array against the
    predicted step function, evaluated on the same t_grid.

    Parameters
    ----------
    t_grid:      array of times (e.g. truth["t"])
    param_true:  array of true values at t_grid (e.g. truth["param_true"])
    pred_taus:   predicted changepoints (list/array, possibly empty)
    pred_values: predicted segment values (len = len(pred_taus) + 1)
    t_min, t_max: domain bounds used to build the predicted step function

    Returns
    -------
    dict with mean_rel_err, median_rel_err, max_rel_err, rmse
    """
    t_grid = np.asarray(t_grid, dtype=float)
    param_true = np.asarray(param_true, dtype=float)

    pred_func = _build_step_func(pred_taus, pred_values, t_min, t_max)
    pred_arr = pred_func(t_grid)

    rel_err = np.abs(pred_arr - param_true) / (np.abs(param_true) + 1e-12)

    return {
        "mean_rel_err": float(np.mean(rel_err)),
        "median_rel_err": float(np.median(rel_err)),
        "max_rel_err": float(np.max(rel_err)),
        "rmse": float(np.sqrt(np.mean((pred_arr - param_true) ** 2))),
    }


def changepoint_metrics(true_taus, pred_taus, tol_h=0.25):
    """
    Match predicted changepoints to true changepoints within a tolerance
    window (greedy nearest-match, one-to-one), then report precision,
    recall, F1, and signed segment-count error.

    Parameters
    ----------
    true_taus: array of true interior changepoint times
    pred_taus: array of predicted interior changepoint times
    tol_h:     tolerance (in same time units as taus) for a match to count

    Returns
    -------
    dict with precision, recall, f1, n_true, n_pred, count_error
    """
    true_taus = np.asarray(true_taus, dtype=float)
    pred_taus = np.asarray(pred_taus, dtype=float)

    matched_true = set()
    matched_pred = set()

    # greedy match: for each predicted cp, find nearest unmatched true cp
    # within tolerance
    if len(pred_taus) > 0 and len(true_taus) > 0:
        # sort predicted taus by how close their best match is, so the
        # tightest matches get claimed first (avoids one pred cp stealing
        # a match that a closer pred cp also wanted)
        candidates = []
        for i, tp in enumerate(pred_taus):
            dists = np.abs(true_taus - tp)
            j = int(np.argmin(dists))
            if dists[j] <= tol_h:
                candidates.append((dists[j], i, j))
        candidates.sort(key=lambda x: x[0])

        for _, i, j in candidates:
            if i in matched_pred or j in matched_true:
                continue
            matched_pred.add(i)
            matched_true.add(j)

    tp_count = len(matched_pred)
    n_pred = len(pred_taus)
    n_true = len(true_taus)

    precision = tp_count / n_pred if n_pred > 0 else (1.0 if n_true == 0 else 0.0)
    recall = tp_count / n_true if n_true > 0 else (1.0 if n_pred == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_true": int(n_true),
        "n_pred": int(n_pred),
        "count_error": int(n_pred - n_true),
    }