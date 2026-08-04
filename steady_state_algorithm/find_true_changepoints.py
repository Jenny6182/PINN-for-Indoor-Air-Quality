"""
Ground-truth segment builder: find true changepoints directly from a
generator's Q_true/S_true columns, then build segment objects with the
transient_idx/steady_idx fields estimate_css/estimate_tau/estimate_segment
expect -- bypassing label_steady_significance_v2/cut_segments entirely.

Use this to isolate Stage A (labeling/segmentation) from Stage B
(Css/tau/Q/S estimation): if estimate_segment gives clean results when
fed these true-boundary segments, Stage B is fine and any errors in the
normal pipeline trace back to Stage A's detected boundaries. If results
are still bad here, the problem is in Stage B (estimation) itself.
"""
import csv
import numpy as np


def find_changepoints(values, rtol=1e-6, atol=1e-9):
    """
    Return indices i (i >= 1) where values[i] differs from values[i-1]
    by more than the given relative/absolute tolerance. Intended for
    Q_true/S_true columns, which are exactly piecewise-constant (no
    noise), so tight tolerances cleanly separate real jumps from
    floating-point noise.
    """
    changepoints = []
    prev = values[0]
    for i in range(1, len(values)):
        cur = values[i]
        if abs(cur - prev) > atol + rtol * abs(prev):
            changepoints.append(i)
        prev = cur
    return changepoints


def load_columns(csv_path, time_col="t_hours", true_q_col="Q_true", true_s_col="S_true"):
    t, q, s = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row[time_col]))
            q.append(float(row[true_q_col]))
            s.append(float(row[true_s_col]))
    return t, q, s


class TrueSegment:
    """Minimal container matching what estimate_segment expects from a segment."""
    def __init__(self, start_idx, end_idx, transient_idx, steady_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transient_idx = transient_idx
        self.steady_idx = steady_idx


def build_true_segments(t, boundaries_idx, transient_duration_hours=None, fraction_steady=0.85):
    """
    Build segment objects from true changepoint indices, splitting each
    segment into a transient portion (start) and steady portion (rest).

    boundaries_idx: sorted changepoint indices from find_changepoints,
        WITHOUT the endpoints -- this function adds start=0 and
        end=len(t)-1 itself.

    transient_duration_hours: if given, the first N hours of each
        segment are treated as transient, remainder as steady -- this
        is preferred over fraction_steady when segment durations vary
        a lot (short + long segments shouldn't use the same fraction).
        If None, falls back to fraction_steady.

    fraction_steady: used only if transient_duration_hours is None.
        Fraction of each segment (by point count) treated as steady;
        the remaining leading fraction is treated as transient. This
        is a rough approximation -- we don't have true per-point
        transient/steady labels, only true segment boundaries.
    """
    t = np.asarray(t)
    full_boundaries = [0] + list(boundaries_idx) + [len(t) - 1]

    segments = []
    for i in range(len(full_boundaries) - 1):
        start_idx = full_boundaries[i]
        is_last = (i == len(full_boundaries) - 2)
        end_idx = full_boundaries[i + 1] if is_last else full_boundaries[i + 1] - 1

        if transient_duration_hours is not None:
            t_start = t[start_idx]
            split = start_idx + np.searchsorted(
                t[start_idx:end_idx + 1], t_start + transient_duration_hours
            )
            split = min(split, end_idx + 1)
        else:
            n = end_idx - start_idx + 1
            split = start_idx + int(n * (1 - fraction_steady))

        transient_idx = np.arange(start_idx, split)
        steady_idx = np.arange(split, end_idx + 1)

        segments.append(TrueSegment(start_idx, end_idx, transient_idx, steady_idx))

    return segments


def get_true_segments_for_mode(csv_path, mode="Q", time_col="t_hours",
                                true_q_col="Q_true", true_s_col="S_true",
                                rtol=1e-6, atol=1e-9,
                                transient_duration_hours=0.5, fraction_steady=0.85):
    """
    One-call convenience: load a CSV, find true changepoints for
    whichever column varies (mode="Q" or "S"), and build true segments
    ready to pass into estimate_segment.

    Returns (t, C, segments) -- note C is NOT loaded here (Q_true/S_true
    only tell you where changepoints are, not the measured concentration);
    load C_meas_ppm separately and pass it alongside t/segments to
    estimate_segment.
    """
    t, q, s = load_columns(csv_path, time_col, true_q_col, true_s_col)
    values = q if mode == "Q" else s
    changepoints = find_changepoints(values, rtol=rtol, atol=atol)

    segments = build_true_segments(
        t, changepoints,
        transient_duration_hours=transient_duration_hours,
        fraction_steady=fraction_steady,
    )
    return t, changepoints, segments