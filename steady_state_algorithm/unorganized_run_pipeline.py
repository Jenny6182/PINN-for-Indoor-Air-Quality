"""
Full non-PINN pipeline:
  load CSV -> label steady/transient -> cut segments -> estimate Css, tau -> solve Q, S

Usage:
  python run_pipeline.py --csv data.csv --V 50 --C_out 420 \
      --time_col t_hours --conc_col C_meas_ppm \
      --slope_percentile 25 --min_run 5
"""
import argparse
import numpy as np
import pandas as pd

from steady_state_algorithm.detect_transient_steady import label_steady, smoothed_derivative
from steady_state_algorithm.segment_cutter import cut_segments
from steady_state_algorithm.estimate_params import estimate_segment
from steady_state_algorithm.plotting import *


def refine_changepoints(t, C, segments, fraction=0.5):
    """
    Refine segment boundaries using the rising edge of the derivative.

    fraction:
        0.3 -> earlier boundary
        0.5 -> middle of derivative rise
        0.7 -> closer to peak

    Uses derivative magnitude threshold instead of peak.
    """

    dCdt = smoothed_derivative(t, C)

    for i in range(1, len(segments)):
        seg = segments[i]

        if len(seg.transient_idx) == 0:
            continue

        transient_idx = np.array(seg.transient_idx)

        # derivative magnitude during transient
        slope = np.abs(dCdt[transient_idx])

        peak = np.max(slope)

        # find first point where derivative reaches fraction of peak
        candidates = transient_idx[
            slope >= fraction * peak
        ]

        if len(candidates) > 0:
            old_idx = seg.start_idx
            new_idx = candidates[0]

            print(
                f"Segment {i}: "
                f"{t[old_idx]:.3f} -> {t[new_idx]:.3f}"
            )

            # update refined changepoint
            seg.start_idx = new_idx

            # points before the refined changepoint belong to previous steady tail
            # but remain inside this segment for visualization purposes
            old_steady_tail = list(range(old_idx, new_idx))

            # rebuild indices after refinement
            seg.transient_idx = list(
                range(new_idx, seg.end_idx + 1)
            )

            # classify the remaining points after transient as steady
            steady_start = seg.end_idx + 1
            for j in range(new_idx, seg.end_idx + 1):
                if abs(dCdt[j]) < np.percentile(
                    np.abs(dCdt[new_idx:seg.end_idx + 1]),
                    50
                ):
                    steady_start = j
                    break

            seg.transient_idx = list(
                range(new_idx, steady_start)
            )
            seg.steady_idx = old_steady_tail + list(
                range(steady_start, seg.end_idx + 1)
            )

    return segments


def run(csv_path, V, C_out, time_col="t_hours", conc_col="C_meas_ppm",
        window_length=21, slope_percentile=50, min_run=10,
        true_Q_col=None, true_S_col=None):
    df = pd.read_csv(csv_path)
    t = df[time_col].to_numpy()
    C = df[conc_col].to_numpy()

    steady = label_steady(t, C, window_length=window_length,
                           slope_percentile=slope_percentile, min_run=min_run)
    segments = cut_segments(list(steady)) # the one without addition into transient part

    segments = refine_changepoints(t, C, segments, fraction=0.5)

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

    plot_true_vs_estimated_with_error(result)
    plot_relative_error(result)

    plot_css_debug(t, C, segments, result)

    plot_debug(
        pd.read_csv(args.csv)[args.time_col].to_numpy(),
        pd.read_csv(args.csv)[args.conc_col].to_numpy(),
        steady,
        segments,
        result
    )

    plot_derivative_with_Q(df, t, C, segments)

    # plot_Q_true_with_boundaries(df, t, segments)
    # plot_true_vs_estimated_segments(
    #         df,
    #         t,
    #         C,
    #         segments,
    #         true_Q_col,
    #         true_S_col
    #     )
    # plot_derivative(t, C)
    plot_parameter_estimates(result)
    plot_parameter_error(result)
    plot_true_vs_estimated_scatter(result)


    return result, steady, segments


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/datasets/no_noise_dataset/iaq_varyQ_seg5_seed57.csv", help="input CSV file")
    p.add_argument("--V", type=float, default=100.0, help="zone volume")
    p.add_argument("--C_out", type=float, default=420.0, help="outdoor/inlet concentration, ppm")
    p.add_argument("--time_col", default="t_hours")
    p.add_argument("--conc_col", default="C_meas_ppm")
    p.add_argument("--window_length", type=int, default=21)
    p.add_argument("--slope_percentile", type=float, default=50)
    p.add_argument("--min_run", type=int, default=10)
    p.add_argument("--true_Q_col", default="Q_true")
    p.add_argument("--true_S_col", default="S_true")
    args = p.parse_args()

    result, steady, segments = run(
        args.csv, args.V, args.C_out, args.time_col, args.conc_col,
        args.window_length, args.slope_percentile, args.min_run,
        args.true_Q_col, args.true_S_col,
    )

    # plots
    compare_Css(result, args.C_out)
    plot_tau_comparison(result, args.V)

    pd.set_option("display.width", 140)
    print(result.to_string(index=False))
