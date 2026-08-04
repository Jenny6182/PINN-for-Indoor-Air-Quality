"""
Full non-PINN pipeline:
  load CSV -> label steady/transient -> cut segments -> refine changepoints
  -> estimate Css, tau per segment -> solve Q, S -> plot results

Usage:
  python run_pipeline.py --csv data.csv --V 50 --C_out 420 \
      --time_col t_hours --conc_col C_meas_ppm \
      --slope_percentile 25 --min_run 5
"""
import argparse
import numpy as np
import pandas as pd

from steady_state_algorithm.detect_transient_steady import *
from steady_state_algorithm.segment_cutter import _build_segment, cut_segments
from steady_state_algorithm.estimate_params import *
from steady_state_algorithm.plotting import *
from core.utils.preprocessing import load_columns
from steady_state_algorithm.find_true_changepoints import *



# def refine_changepoints(t, C, segments, fraction=0.5):
#     """
#     Refine segment boundaries using the rising edge of the derivative,
#     rather than trusting the initial slope-threshold cut exactly.

#     For each segment (after the first), finds where |dC/dt| first
#     reaches `fraction` of its peak value within that segment's
#     transient region, and moves the segment's start there. 
    
#     After moving the boundary, points are reclassified into
#     "transient" (from the new start until the derivative drops back
#     below the median for that segment) and "steady" (everything else,
#     including the old pre-refinement tail).

#     fraction: how far up the derivative's rise to place the boundary.
#         0.3 -> earlier boundary, 0.5 -> middle of the rise, 0.7 -> closer
#         to the peak.
    
#     From observations, it seems usually the changepoint is at 0.5 of transient detected
#     """
#     dCdt = smoothed_derivative(t, C)

#     for i in range(1, len(segments)):
#         seg = segments[i]
#         if len(seg.transient_idx) == 0:
#             continue

#         transient_idx = np.array(seg.transient_idx)
#         slope = np.abs(dCdt[transient_idx])
#         peak = np.max(slope)

#         # first point where derivative reaches fraction of peak
#         candidates = transient_idx[slope >= fraction * peak]
#         if len(candidates) == 0:
#             continue

#         old_idx = seg.start_idx
#         new_idx = candidates[0]
#         print(f"Segment {i}: {t[old_idx]:.3f} -> {t[new_idx]:.3f}")

#         seg.start_idx = new_idx

#         # points before the refined changepoint belong to the previous
#         # segment's steady tail, but stay in this segment's index lists
#         # for visualization purposes
#         old_steady_tail = list(range(old_idx, new_idx))

#         # re-split [new_idx, seg.end_idx] into transient vs. steady:
#         # transient runs until the derivative drops below the median
#         # for that range, everything after that is steady
#         steady_start = seg.end_idx + 1
#         median_slope = np.percentile(np.abs(dCdt[new_idx:seg.end_idx + 1]), 50)
#         for j in range(new_idx, seg.end_idx + 1):
#             if abs(dCdt[j]) < median_slope:
#                 steady_start = j
#                 break

#         seg.transient_idx = list(range(new_idx, steady_start))
#         seg.steady_idx = old_steady_tail + list(range(steady_start, seg.end_idx + 1))

#     return segments


def force_leading_transient(steady):
    """
    The very start of a trace can never legitimately be in steady
    state -- there's no prior state to have converged from. Curvature
    bias in the local slope significance test (a short forward-only
    window at t=0 averages the steep early rise with a flatter later
    stretch, underestimating the true instantaneous slope) can still
    mislabel the first few points as "steady" despite this.

    Fix: find the first point the test already correctly labeled
    transient, and force everything before it to transient too. Only
    touches the very beginning of the array (index 0 onward) -- later
    segment transitions aren't affected, since they have real prior
    data behind them and don't show the same bias.
    """
    steady = steady.copy()
    transient_indices = np.where(~steady)[0]
    if len(transient_indices) == 0:
        return steady  # no transient detected anywhere; nothing to force
    first_transient = transient_indices[0]
    steady[:first_transient] = False
    return steady


def run(csv_path, V, C_out, time_col="t_hours", conc_col="C_meas_ppm",
        window_length=21, min_steady_run=10, min_transient_run=8,
        sigma_meas=15.0, crit=1.5,
        true_Q_col=None, true_S_col=None, debug=True):
    """
    Run the full pipeline on one CSV: label steady/transient points,
    cut segments, estimate Css/tau/Q/S per segment, print a
    per-segment comparison against ground truth (if true_Q_col /
    true_S_col are given), and generate the full set of diagnostic plots.

    If debug=True, also plots a stage-by-stage comparison of the
    steady/transient labeling (raw significance test, post-filtering,
    and final per-segment indices) so labeling issues can be traced to
    the specific stage that introduced them.

    Returns (result, steady, segments):
      result   -- DataFrame, one row per segment with est. + true values
      steady   -- boolean array, steady/transient label per timestep
      segments -- list of segment objects
    """
    df = pd.read_csv(csv_path)
    t = df[time_col].to_numpy()
    C = df[conc_col].to_numpy()

    steady = label_steady_significance_v2(
        t, C, window=window_length, sigma_meas=sigma_meas, crit=crit,
        min_steady_run=min_steady_run, min_transient_run=min_transient_run,
    )

    steady = force_leading_transient(steady)

    _, q, s = load_columns(csv_path, time_col, true_Q_col, true_S_col)
    mode_values = q if true_Q_col is not None else s
    changepoints = find_changepoints(mode_values)

    segments = build_true_segments(t, changepoints, transient_duration_hours=0.5)

    # segments = cut_segments(list(steady))  # initial cut, before changepoint refinement
    # segments = refine_changepoints(t, C, segments, fraction=0.5)  # not compatible with noisy data yet

    if debug:
        t_stat = local_slope_significance(t, C, window=window_length, sigma_meas=sigma_meas)
        raw_steady = np.nan_to_num(np.abs(t_stat) < crit, nan=False).astype(bool)

        final_steady = np.zeros_like(steady)
        for seg in segments:
            final_steady[seg.steady_idx] = True

        true_change_times = None
        if true_Q_col is not None:
            Q_arr = df[true_Q_col].to_numpy()
            change_idx = np.where(Q_arr[1:] != Q_arr[:-1])[0] + 1
            true_change_times = t[change_idx]

        print("PLOTTING STAGE COMPARISON...")
        plot_stage_comparison(t, C, {
            "raw t-stat, no filtering": raw_steady,
            "after v2 run-length filtering": steady,
            "final: seg.steady_idx / seg.transient_idx": final_steady,
        }, true_change_times=true_change_times)

    rows = []
    for i, seg in enumerate(segments):
        est = estimate_segment(t, C, seg, V=V, C_out=C_out)

        row = {
            "t_start": t[seg.start_idx],
            "t_end": t[seg.end_idx],
            "n_steady": est.n_steady,
            "n_transient": est.n_transient,
            "Css_est": est.Css,
            "tau_est": est.tau,
            "Q_est": est.Q,
            "S_est": est.S,
            "tau_r2": est.tau_fit_r2,
        }

        if true_Q_col is not None:
            Q_true = df[true_Q_col].iloc[seg.start_idx:seg.end_idx + 1].mean()
            row["Q_true"] = Q_true
            row["tau_true"] = V / Q_true

        if true_S_col is not None:
            S_true = df[true_S_col].iloc[seg.start_idx:seg.end_idx + 1].mean()
            row["S_true"] = S_true

        if true_Q_col is not None and true_S_col is not None:
            row["Css_true"] = C_out + S_true / Q_true

        rows.append(row)

        print(f"\nSegment {i}")
        print(f"  Css : est = {row['Css_est']:.4f},  true = {row.get('Css_true', np.nan):.4f}")
        print(f"  tau : est = {row['tau_est']:.4f},  true = {row.get('tau_true', np.nan):.4f}")
        print(f"  Q   : est = {row['Q_est']:.4f},  true = {row.get('Q_true', np.nan):.4f}")
        print(f"  S   : est = {row['S_est']:.4f},  true = {row.get('S_true', np.nan):.4f}")

    result = pd.DataFrame(rows)

    # diagnostic plots
    plot_css_debug(t, C, segments, result)
    plot_derivative_with_Q(df, t, C, segments)
    plot_true_vs_estimated_with_error(result)
    plot_relative_error(result)

    plot_Q_true_with_boundaries(df, t, segments)
    plot_true_vs_estimated_segments(df, t, C, segments, true_Q_col, true_S_col)
    plot_true_vs_estimated_scatter(result)

    return result, steady, segments


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/datasets/validation_dataset/iaq_varyQ_seg5_seed57.csv", help="input CSV file")
    p.add_argument("--V", type=float, default=100.0, help="zone volume")
    p.add_argument("--C_out", type=float, default=420.0, help="outdoor/inlet concentration, ppm")
    p.add_argument("--time_col", default="t_hours")
    p.add_argument("--conc_col", default="C_meas_ppm")
    p.add_argument("--window_length", type=int, default=21)
    p.add_argument("--min_steady_run", type=int, default=10)
    p.add_argument("--min_transient_run", type=int, default=8)
    p.add_argument("--true_Q_col", default="Q_true")
    p.add_argument("--true_S_col", default="S_true")
    p.add_argument("--crit", type=float, default=1.5)
    p.add_argument("--sigma_meas", type=float, default=15.0)
    p.add_argument("--no-debug", dest="debug", action="store_false", help="disable stage-by-stage labeling comparison plot")
    args = p.parse_args()

    result, steady, segments = run(
        csv_path=args.csv,
        V=args.V,
        C_out=args.C_out,
        time_col=args.time_col,
        conc_col=args.conc_col,
        window_length=args.window_length,
        min_steady_run=args.min_steady_run,
        min_transient_run=args.min_transient_run,
        sigma_meas=args.sigma_meas,
        crit=args.crit,
        true_Q_col=args.true_Q_col,
        true_S_col=args.true_S_col,
        debug=args.debug,
    )

    compare_Css(result, args.C_out)
    plot_tau_comparison(result, args.V)

    pd.set_option("display.width", 140)
    print(result.to_string(index=False))