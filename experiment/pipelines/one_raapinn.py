# import sys

# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# from core.utils.preprocessing import prepare_training_data, compute_stats
# from core.scan.window_sweeping import *
# from experiment.pipelines.stage2 import run_stage2
# from core.utils.plotting import plot_all_raa, plot_stage2_interval
# from core.pinn.collocation import to_torch
# from experiment.configs.config import V, C_out, SEED
# from core.pinn.collocation import to_torch
# from core.utils.preprocessing import normalize_with_stats
# from datetime import datetime
# import torch
# import numpy as np
# import os

# # Get the current working directory
# current_dir = os.getcwd()
# print(current_dir)



# """

# Preprocess data
# stage1

# create collocation points
# stage2, init a nn with param model, and train that

# """

# # pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
# torch.manual_seed(SEED) # set random numbers in torch
# np.random.seed(SEED) # in numpy


# def main(path="./data/datasets/varying_pinn_datasets/iaq_co2_varying_Q.csv", k=None, prominence_factor=0.15):



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
from core.scan.window_sweeping import *
from experiment.pipelines.one_stage2 import run_one_stage2
from core.utils.plotting import plot_all_raa, plot_stage2_interval
from core.pinn.collocation import to_torch
from experiment.configs.config import V, C_out, SEED
from core.pinn.collocation import to_torch
from core.utils.preprocessing import normalize_with_stats
from datetime import datetime
import torch
import numpy as np

# Hyperparams for stage I
WINDOW_SIZE  = 20      # points per sliding window (~20 min at 1-min sampling)
SIGMA        = 1.5     # Gaussian smoothing sigma before differentiation
PROMINENCE   = None    # set to None to auto-set as 15% of max score
# prominence_factor = 0.05 #default 0.15
DISTANCE     = 20      # min points between peaks (same as WINDOW_SIZE is safe)
MARGIN_H     = 0.4     # candidate interval half-width around each peak [hours]

# ----- SET SEED -----
# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
torch.manual_seed(SEED) # set random numbers in torch
np.random.seed(SEED) # in numpy


def main(path="./data/datasets/varying_pinn_datasets/varying_Q.csv", 
         log_Q_init=np.log(200.0),
         log_S_init=np.log(1e5),
         k=None, prominence_factor=0.15,
         run_dir=None):

    stem = Path(path).stem  # extract filename without extension early

    # If run_dir not provided, create one
    if run_dir is None:
        run_dir = Path(f"results/sensitivity_analysis/{stem}_Q{log_Q_init}_S{log_S_init}")
    else:
        run_dir = Path(run_dir)
    
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"RAA-PINN Pipeline")
    print(f"File: {path}")
    print(f"{'='*60}")

    # ----- data preprocessing ----- 
    # extra_cols loads the ground truth Q and S columns for evaluation
    data = prepare_training_data(path, x_col="t_hours", y_col="C_meas_ppm", extra_cols=["Q_true", "S_true"],)

    t_np      = data["t_np"]        # shape (N, 1)
    C_meas_np = data["c_np"]        # shape (N, 1)
    Q_true_np = data["Q_true"]      # shape (N,)
    S_true_np = data["S_true"]      # shape (N,)

    print(f"\nLoaded {len(t_np)} timepoints  "
          f"(t = {t_np.min():.2f}h to {t_np.max():.2f}h)")

    # ----- Stage I ----- 
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

    if k is not None:
        print("k: ", k)
        peak_indices, intervals = find_top_k_intervals(
            t=t_np,
            scores=scores,
            k=k,
            margin_h=MARGIN_H,
        )
        print(f"  Using top-{k} peaks (known changepoint count)")
    else:
        prominence = PROMINENCE
        if prominence is None:
            prominence = prominence_factor * np.nanmax(scores)
            print(f"  Auto prominence = {prominence:.3e} (15% of max score)")
            # auto-set prominence if not specified
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
              f"-> interval [{tl:.3f}h, {tr:.3f}h]")

    if len(peak_indices) == 0:
        print("\n  No changepoints detected. Try lowering PROMINENCE or DISTANCE.")
        return

    t = t_np.flatten()      # shape (N, 1)
    tau_inits = [t[i] for i in peak_indices]
    t_min = t.min()
    t_max = t.max()

    # ----- Stage II -----
    print(f"\n--- Stage II: Joint Refinement (One PINN) ---")

    stage2_result = run_one_stage2(
        t_np=t_np,
        C_meas_np=C_meas_np,
        tau_inits=tau_inits,
        t_min=t_min,
        t_max=t_max,
        log_Q_init=log_Q_init,
        log_S_init=log_S_init,
        print_every=500,
        verbose=True,
    )

    # ----- print summary table -----
    print(f"\n{'='*60}")
    print(f"Summary: {len(stage2_result['taus'])} changepoints detected and refined")
    print(f"{'='*60}")
    print(f"  {'#':>3}  {'tau_est':>8}  {'Q_minus':>8}  {'Q_plus':>8}  "
          f"{'S_minus':>10}  {'S_plus':>10}")
    print(f"  {'-'*60}")
    for i, (tau, Q_m, Q_p, S_m, S_p) in enumerate(
        zip(stage2_result["taus"], stage2_result["Q_minus"], stage2_result["Q_plus"],
            stage2_result["S_minus"], stage2_result["S_plus"])):
        print(f"  {i+1:>3}  {tau:>8.3f}  {Q_m:>8.1f}  "
              f"{Q_p:>8.1f}  {S_m:>10.2e}  {S_p:>10.2e}")

    # compare against true values at each detected changepoint
    print(f"\n  True Q at each detected changepoint:")
    for i, tau in enumerate(stage2_result["taus"]):
        idx_before = np.searchsorted(t, tau) - 1
        idx_after  = np.searchsorted(t, tau)
        idx_before = np.clip(idx_before, 0, len(t) - 1)
        idx_after  = np.clip(idx_after,  0, len(t) - 1)
        print(f"    [{i+1}] tau={tau:.3f}h  "
              f"Q_true before={Q_true_np[idx_before]:.1f}  "
              f"Q_true after={Q_true_np[idx_after]:.1f}  |  "
              f"estimated Q-={stage2_result['Q_minus'][i]:.1f}  Q+={stage2_result['Q_plus'][i]:.1f}")

    # ----- main RAA diagnostic plot -----
    # Convert stage2_result arrays into the format expected by plot_all_raa
    stage2_results_for_plot = []
    for i, tau in enumerate(stage2_result["taus"]):
        stage2_results_for_plot.append({
            "tau": tau,
            "Q_minus": stage2_result["Q_minus"][i],
            "Q_plus": stage2_result["Q_plus"][i],
            "S_minus": stage2_result["S_minus"][i],
            "S_plus": stage2_result["S_plus"][i],
        })
    
    output_path = run_dir / "raa_diagnostic.png"

    plot_all_raa(
        t_np=t_np,
        c_np=C_meas_np,
        scores=scores,
        peak_indices=peak_indices,
        stage2_results=stage2_results_for_plot,
        Q_true_np=Q_true_np,
        S_true_np=S_true_np,
        output_path=output_path,
    )

    # return stage2_result
    results = []

    for tau, Q_m, Q_p, S_m, S_p in zip(
        stage2_result["taus"],
        stage2_result["Q_minus"],
        stage2_result["Q_plus"],
        stage2_result["S_minus"],
        stage2_result["S_plus"],
    ):
        results.append({
            "tau": tau,
            "Q_minus": Q_m,
            "Q_plus": Q_p,
            "S_minus": S_m,
            "S_plus": S_p,
        })

    return results


if __name__ == "__main__":
    # run on varying Q dataset by default
    # change the path here to run on varying S
    if len(sys.argv) < 5:
        results = main()
    else:
        dataset_path = sys.argv[1]
        print(dataset_path)
        log_Q_init = float(sys.argv[2])
        print(log_Q_init)
        log_S_init = float(sys.argv[3])
        print(log_S_init)
        k = int(sys.argv[4]) if sys.argv[4] != "None" else None
        print(k)
        prominence_factor = float(sys.argv[4])
        print(prominence_factor)
        
        results = main(dataset_path, log_Q_init, log_S_init, k, prominence_factor)

    print(results.keys())