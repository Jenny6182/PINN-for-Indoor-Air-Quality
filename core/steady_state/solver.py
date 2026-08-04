"""
core/steady_state/solver.py
-----------------------------
Given steady-state points pooled from each segment, solve directly (no
training) for the per-segment varying parameter and the single shared
constant, using the same linear relation stage1_scan already relies on:

    V * dC/dt = Q * (C_out - C) + S

At steady state dC/dt ~= 0, but rather than treating that as a single
equation per segment (which underdetermines the shared constant), we pool
ALL steady points from ALL segments into one joint least-squares problem:
one unknown per segment for the varying parameter, plus one shared unknown
for the constant parameter. This mirrors stage1_scan's own per-window
lstsq fit, just solved once globally instead of locally.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import lstsq


def solve_joint_steady_state(
    segment_steady_data: list[dict],
    V: float,
    C_out: float,
    vary_param: str,
) -> dict:
    """
    Params
    ------
    segment_steady_data : list of dicts, one per segment (in tau order),
        each with keys:
            "C_steady"    : array of C values at steady-state points in this segment
            "dCdt_steady" : array of dC/dt values at those same points
        A segment with zero steady points should still have an entry with
        empty arrays -- it contributes no rows, but keeps segment indexing
        consistent.
    vary_param : "Q" or "S" -- which parameter varies per segment. The other
        is solved as a single shared constant across all segments.

    Returns
    -------
    dict with:
        "varying"  : array of length K+1, the per-segment estimate of the
                     varying parameter (Q if vary_param=="Q", else S)
        "constant" : float, the shared estimate of the other parameter
        "n_steady_points": array of length K+1, how many steady points fed
                     into each segment's estimate (0 means that segment's
                     "varying" entry is unconstrained -- flag it)
    """
    assert vary_param in ("Q", "S")
    K1 = len(segment_steady_data)

    rows = []
    b = []
    n_points = np.zeros(K1, dtype=int)

    for i, seg in enumerate(segment_steady_data):
        C_s = np.asarray(seg["C_steady"], dtype=np.float64)
        dCdt_s = np.asarray(seg["dCdt_steady"], dtype=np.float64)
        n_points[i] = len(C_s)

        for C_val, dCdt_val in zip(C_s, dCdt_s):
            row = np.zeros(K1 + 1)
            if vary_param == "Q":
                row[i] = (C_out - C_val)   # coefficient on this segment's Q
                row[-1] = 1.0              # coefficient on shared S
            else:  # vary_param == "S"
                row[i] = 1.0               # coefficient on this segment's S
                row[-1] = (C_out - C_val)  # coefficient on shared Q
            rows.append(row)
            b.append(V * dCdt_val)

    if not rows:
        raise ValueError(
            "No steady-state points found in any segment -- cannot solve. "
            "Check classify_steady_transient's eps_rel/min_duration_frac, "
            "or this dataset may not have a clean steady-state scenario."
        )

    A = np.array(rows)
    b = np.array(b)

    params, _, _, _ = lstsq(A, b)

    varying = params[:K1]
    constant = params[-1]

    zero_point_segments = np.where(n_points == 0)[0]
    if len(zero_point_segments) > 0:
        print(
            f"WARNING: segments {zero_point_segments.tolist()} had 0 steady "
            f"points -- their estimate is effectively unconstrained by data "
            f"(solved only via the shared constant's influence, if any)."
        )

    return {
        "varying": varying,
        "constant": float(constant),
        "n_steady_points": n_points,
    }