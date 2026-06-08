"""
raa_pinn_main.py
----------------
RAA-PINN pipeline for unknown changepoint detection and parameter estimation.

Implements a cheaper version of the RAA-PINNs paper (Bai et al. 2026) that
exploits the linearity of the IAQ ODE to replace Stage I's expensive per-window
PINNs with simple least squares fits.

Pipeline
--------
Stage I  — sliding window least squares scan over the full time series.
           Produces a residual score at every timepoint. Peaks in the score
           indicate where Q or S probably jumped. No neural network involved.

Stage II — for each detected peak, train a small PINN on the short candidate
           interval around that peak. The PINN has a sigmoid-parametrised
           changepoint that is jointly optimised with Q_minus, Q_plus, S_minus,
           S_plus. Produces an exact tau estimate and parameter values on each
           side of the jump.

Output   — piecewise Q(t) and S(t) step functions, one value per detected
           segment, compared against true values from the CSV.

Usage
-----
Run directly:
    python raa_pinn_main.py

Or call main() with a different CSV path:
    from raa_pinn_main import main
    main("varying_datasets/iaq_co2_varying_S.csv")

Key tuning parameters (top of this file):
    WINDOW_SIZE   — width of sliding window in number of timepoints.
                    Should be ~2x the fastest time constant in your data.
                    If peaks are too noisy, increase. If peaks are too wide, decrease.

    SIGMA         — Gaussian smoothing applied before differentiation.
                    Increase if derivative is too noisy. Decrease if peaks are blurred.

    PROMINENCE    — minimum prominence for peak detection.
                    Start at 10-20% of max score, increase to remove false positives,
                    decrease if real changepoints are missed.

    DISTANCE      — minimum timepoints between two detected peaks.
                    Set to roughly WINDOW_SIZE to avoid double-detecting one changepoint.

    MARGIN_H      — half-width of candidate interval passed to Stage II [hours].
                    True changepoint must lie within peak_time +/- MARGIN_H.
                    Use ~1-2x SEGMENT_DURATION.
"""

import os

# Get the current working directory
current_dir = os.getcwd()

print(current_dir)

import sys

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.utils.preprocessing import prepare_training_data, compute_stats
from stage1 import stage1_scan, find_candidate_intervals
from stage2 import run_stage2
from core.utils.plotting import plot_all_raa, plot_stage2_interval
from core.pinn.collocation import to_torch
from configs.config import V, C_out, SEED
from core.pinn.collocation import to_torch
from core.utils.preprocessing import normalize_with_stats

import torch
import numpy as np

# ----- SET SEED -----
# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
torch.manual_seed(SEED) # set random numbers in torch
np.random.seed(SEED) # in numpy


# ── Stage I tuning ────────────────────────────────────────────────────────────
WINDOW_SIZE  = 20      # points per sliding window (~20 min at 1-min sampling)
SIGMA        = 1.5     # Gaussian smoothing sigma before differentiation
PROMINENCE   = None    # set to None to auto-set as 15% of max score
# prominence_factor = 0.05 #default 0.15
DISTANCE     = 20      # min points between peaks (same as WINDOW_SIZE is safe)
MARGIN_H     = 0.4     # candidate interval half-width around each peak [hours]


def main(path="varying_datasets/iaq_co2_varying_Q.csv", prominence_factor=0.15):

    print(f"\n{'='*60}")
    print(f"RAA-PINN Pipeline")
    print(f"File: {path}")
    print(f"{'='*60}")

    # ── load data ─────────────────────────────────────────────────────────────
    # extra_cols loads the ground truth Q and S columns for evaluation
    data = prepare_training_data(
        path,
        x_col="t_hours",
        y_col="C_meas_ppm",
        extra_cols=["Q_true", "S_true"],
    )

    t_np      = data["t_np"]        # shape (N, 1)
    C_meas_np = data["c_np"]        # shape (N, 1)
    Q_true_np = data["Q_true"]      # shape (N,)
    S_true_np = data["S_true"]      # shape (N,)

    print(f"\nLoaded {len(t_np)} timepoints  "
          f"(t = {t_np.min():.2f}h to {t_np.max():.2f}h)")

    # ── Stage I ───────────────────────────────────────────────────────────────
    print(f"\n--- Stage I: Sliding Window Scan ---")
    print(f"  window_size={WINDOW_SIZE}, sigma={SIGMA}")

    scores = stage1_scan(
        t=t_np,
        C_meas=C_meas_np,
        V=V,
        C_out=C_out,
        window_size=WINDOW_SIZE,
        sigma=SIGMA,
    )

    # auto-set prominence if not specified
    prominence = PROMINENCE
    if prominence is None:
        prominence = prominence_factor * np.nanmax(scores)
        print(f"  Auto prominence = {prominence:.3e}  (15% of max score)")

    peak_indices, intervals = find_candidate_intervals(
        t=t_np,
        scores=scores,
        prominence=prominence,
        distance=DISTANCE,
        margin_h=MARGIN_H,
    )

    print(f"\n  Detected {len(peak_indices)} candidate changepoints:")
    t_flat = t_np.flatten()
    for i, (idx, (tl, tr)) in enumerate(zip(peak_indices, intervals)):
        print(f"    [{i+1}] peak at t={t_flat[idx]:.3f}h  "
              f"→ interval [{tl:.3f}h, {tr:.3f}h]")

    if len(peak_indices) == 0:
        print("\n  No changepoints detected. Try lowering PROMINENCE or DISTANCE.")
        return

    # ── Stage II ──────────────────────────────────────────────────────────────
    print(f"\n--- Stage II: Joint Refinement ---")

    stage2_results = []
    for i, (t_left, t_right) in enumerate(intervals):
        print(f"\n  Changepoint {i+1}/{len(intervals)}")
        result = run_stage2(
            t_np=t_np,
            C_meas_np=C_meas_np,
            t_left=t_left,
            t_right=t_right,
            print_every=500,
            verbose=True,
        )
        stage2_results.append(result)

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  {'#':>3}  {'tau_est':>8}  {'Q_minus':>8}  {'Q_plus':>8}  "
          f"{'S_minus':>10}  {'S_plus':>10}")
    print(f"  {'-'*60}")
    for i, r in enumerate(stage2_results):
        print(f"  {i+1:>3}  {r['tau']:>8.3f}  {r['Q_minus']:>8.1f}  "
              f"{r['Q_plus']:>8.1f}  {r['S_minus']:>10.2e}  {r['S_plus']:>10.2e}")

    # compare against true values at each detected changepoint
    print(f"\n  True Q at each detected changepoint:")
    for i, r in enumerate(stage2_results):
        tau = r["tau"]
        # find the true Q value just before and after tau
        idx_before = np.searchsorted(t_flat, tau) - 1
        idx_after  = np.searchsorted(t_flat, tau)
        idx_before = np.clip(idx_before, 0, len(t_flat) - 1)
        idx_after  = np.clip(idx_after,  0, len(t_flat) - 1)
        print(f"    [{i+1}] tau={tau:.3f}h  "
              f"Q_true before={Q_true_np[idx_before]:.1f}  "
              f"Q_true after={Q_true_np[idx_after]:.1f}  |  "
              f"estimated Q-={r['Q_minus']:.1f}  Q+={r['Q_plus']:.1f}")

    # ── Stage II interval plots ───────────────────────────────────────────────
    # one subplot per detected changepoint showing the fit on that interval
    if len(stage2_results) > 0:
        n_intervals = len(stage2_results)
        fig, axes   = plt.subplots(1, n_intervals,
                                   figsize=(5 * n_intervals, 4),
                                   squeeze=False)
        for i, r in enumerate(stage2_results):
            plot_stage2_interval(
                ax=axes[0][i],
                model=r["model"],
                t_interval_np=r["t_interval"],
                c_interval_np=r["C_interval"],
                tau_est=r["tau"],
                stats=r["stats"],
                norm_t=normalize_with_stats,
                interval_label=f"changepoint {i+1}",
            )
        fig.suptitle("Stage II — Interval Fits", fontsize=12)
        fig.tight_layout()
        stem     = Path(path).stem
        fig_path = f"iaq_raa_stage2_intervals_{stem}.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        print(f"\nSaved interval plots → {fig_path}")
        plt.close(fig)

    # ── main RAA diagnostic plot ──────────────────────────────────────────────
    stem        = Path(path).stem
    output_path = f"factor={prominence_factor}_raa_diagnostic.png"

    plot_all_raa(
        t_np=t_np,
        c_np=C_meas_np,
        scores=scores,
        peak_indices=peak_indices,
        stage2_results=stage2_results,
        Q_true_np=Q_true_np,
        S_true_np=S_true_np,
        output_path=output_path,
    )

    return stage2_results


if __name__ == "__main__":
    # run on varying Q dataset by default
    # change the path here to run on varying S
    dataset_path = sys.argv[1]
    print(dataset_path)
    prominence_factor = float(sys.argv[2])
    print(prominence_factor)
    results = main(dataset_path, prominence_factor)
    # results = main("./general_pinn/iaq_co2_varying_Q.csv", )