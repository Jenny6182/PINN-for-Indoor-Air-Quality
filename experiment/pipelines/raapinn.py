"""
RAA-PINN Pipeline
-----------------
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
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from core.utils.preprocessing import prepare_training_data, normalize_with_stats
from core.scan.window_sweeping import (
    stage1_scan, find_candidate_intervals, find_top_k_intervals
)
from experiment.pipelines.stage2 import run_stage2
from experiment.pipelines.base import BasePipeline
from core.utils.plotting import plot_all_raa, plot_stage2_interval
from experiment.configs.config import V, C_out, SEED

# ----- SET SEED -----
torch.manual_seed(SEED)
np.random.seed(SEED)


class RAAPINNPipeline(BasePipeline):
    """
    RAA-PINN: Rapid Anomaly detection with Adaptive PINNs.
    Detects unknown changepoints via sliding window scan, then trains one PINN per interval.
    """

    @property
    def model_type(self) -> str:
        return "raapinn"

    @classmethod
    def get_default_config(cls) -> dict:
        """Default hyperparameters for RAA-PINN."""
        return {
            # Stage I: Sliding window scan
            "window_size": 20,           # points per sliding window
            "sigma": 1.5,                # Gaussian smoothing sigma
            "prominence_factor": 0.15,   # prominence as % of max score
            "distance": 20,              # min points between peaks
            "margin_h": 0.4,             # candidate interval half-width [hours]
            
            # Stage II: Per-interval PINN training
            "log_Q_init": np.log(200.0),
            "log_S_init": np.log(1e5),
            "k": None,  # if set, use top-k peaks instead of prominence-based detection
        }

    def run(self, data_path: str, **kwargs) -> dict:
        """
        Execute RAA-PINN pipeline.

        Parameters
        ----------
        data_path : str
            Path to CSV file with columns: t_hours, C_meas_ppm, Q_true, S_true
        **kwargs : dict
            Overrides for default config (log_Q_init, log_S_init, window_size, etc.)

        Returns
        -------
        dict
            Results with keys: stage2_results (list of dicts), peak_indices, intervals
        """
        # Merge config with kwargs
        config = {**self.config.get("raapinn", {}), **kwargs}

        stem = Path(data_path).stem
        print(f"\nRAA-PINN Pipeline")
        print(f"File: {data_path}")
        print(f"Run directory: {self.run_dir}")
        print(f"{'='*60}")

        # ----- Data preprocessing -----
        data = prepare_training_data(
            data_path,
            x_col="t_hours",
            y_col="C_meas_ppm",
            extra_cols=["Q_true", "S_true"],
        )

        t_np = data["t_np"]  # shape (N, 1)
        C_meas_np = data["c_np"]  # shape (N, 1)
        Q_true_np = data["Q_true"]  # shape (N,)
        S_true_np = data["S_true"]  # shape (N,)

        print(f"\nLoaded {len(t_np)} timepoints "
              f"(t = {t_np.min():.2f}h to {t_np.max():.2f}h)")

        # ----- Stage I: Sliding Window Scan -----
        print(f"\n--- Stage I: Sliding Window Scan ---")
        print(f"  window_size={config['window_size']}, sigma={config['sigma']}")

        scores = stage1_scan(
            t=t_np,
            C_meas=C_meas_np,
            V=V,
            C_out=C_out,
            window_size=config["window_size"],
            sigma=config["sigma"],
        )

        if config["k"] is not None:
            print(f"  Using top-{config['k']} peaks (known changepoint count)")
            peak_indices, intervals = find_top_k_intervals(
                t=t_np,
                scores=scores,
                k=config["k"],
                margin_h=config["margin_h"],
            )
        else:
            prominence = config["prominence_factor"] * np.nanmax(scores)
            print(f"  Auto prominence = {prominence:.3e} "
                  f"({config['prominence_factor']*100:.0f}% of max score)")
            peak_indices, intervals = find_candidate_intervals(
                t=t_np,
                scores=scores,
                prominence=prominence,
                distance=config["distance"],
                margin_h=config["margin_h"],
            )

        print(f"\n  Detected {len(peak_indices)} candidate changepoints:")
        t_flat = t_np.flatten()
        for i, (idx, (tl, tr)) in enumerate(zip(peak_indices, intervals)):
            print(f"    [{i+1}] peak at t={t_flat[idx]:.3f}h "
                  f"-> interval [{tl:.3f}h, {tr:.3f}h]")

        if len(peak_indices) == 0:
            print("\n  No changepoints detected. Try lowering prominence or distance.")
            return {
                "stage2_results": [],
                "peak_indices": peak_indices,
                "intervals": intervals,
            }

        # ----- Stage II: Joint Refinement -----
        print(f"\n--- Stage II: Joint Refinement (one PINN per interval) ---")

        stage2_results = []
        for i, (t_left, t_right) in enumerate(intervals):
            print(f"\n  Changepoint {i+1}/{len(intervals)}")
            result = run_stage2(
                t_np=t_np,
                C_meas_np=C_meas_np,
                t_left=t_left,
                t_right=t_right,
                log_Q_init=config["log_Q_init"],
                log_S_init=config["log_S_init"],
                print_every=500,
                verbose=True,
            )
            stage2_results.append(result)

        # ----- Print summary table -----
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"  {'#':>3}  {'tau_est':>8}  {'Q_minus':>8}  {'Q_plus':>8}  "
              f"{'S_minus':>10}  {'S_plus':>10}")
        print(f"  {'-'*60}")
        for i, r in enumerate(stage2_results):
            print(f"  {i+1:>3}  {r['tau']:>8.3f}  {r['Q_minus']:>8.1f}  "
                  f"{r['Q_plus']:>8.1f}  {r['S_minus']:>10.2e}  {r['S_plus']:>10.2e}")

        # Compare against true values at each detected changepoint
        print(f"\n  True Q at each detected changepoint:")
        for i, r in enumerate(stage2_results):
            tau = r["tau"]
            idx_before = np.clip(np.searchsorted(t_flat, tau) - 1, 0, len(t_flat) - 1)
            idx_after = np.clip(np.searchsorted(t_flat, tau), 0, len(t_flat) - 1)
            print(f"    [{i+1}] tau={tau:.3f}h "
                  f"Q_true before={Q_true_np[idx_before]:.1f} "
                  f"Q_true after={Q_true_np[idx_after]:.1f} | "
                  f"estimated Q-={r['Q_minus']:.1f} Q+={r['Q_plus']:.1f}")

        # ----- Save Stage II interval plots -----
        if len(stage2_results) > 0:
            n_intervals = len(stage2_results)
            fig, axes = plt.subplots(
                1, n_intervals,
                figsize=(5 * n_intervals, 4),
                squeeze=False
            )
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
            fig.suptitle("Stage II -> Interval Fits", fontsize=12)
            fig.tight_layout()
            fig_path = self.run_dir / f"iaq_raa_stage2_intervals_{stem}.png"
            fig.savefig(fig_path, dpi=130, bbox_inches="tight")
            print(f"\nSaved interval plots -> {fig_path}")
            plt.close(fig)

        # ----- Save main RAA diagnostic plot -----
        output_path = self.run_dir / "raa_diagnostic.png"
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

        print(f"\n{'='*60}")
        print(f"Pipeline complete")
        print(f"{'='*60}\n")

        return {
            "stage2_results": stage2_results,
            "peak_indices": peak_indices,
            "intervals": intervals,
            "t_np": t_np,
            "C_meas_np": C_meas_np,
            "scores": scores,
        }


# Legacy interface for backward compatibility
def main(path="./data/datasets/varying_pinn_datasets/iaq_co2_varying_Q.csv",
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

    # ----- Stage II -----
    print(f"\n--- Stage II: Joint Refinement ---")

    stage2_results = []
    for i, (t_left, t_right) in enumerate(intervals):
        print(f"\n  Changepoint {i+1}/{len(intervals)}")
        result = run_stage2(
            t_np=t_np,
            C_meas_np=C_meas_np,
            t_left=t_left,
            t_right=t_right,
            log_Q_init=log_Q_init,
            log_S_init=log_S_init,
            print_every=500,
            verbose=True,
        )
        stage2_results.append(result)

    # ----- print summary table -----
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

    # ----- Stage II interval plots -----
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
        fig_path = run_dir / f"iaq_raa_stage2_intervals_{stem}.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        print(f"\nSaved interval plots -> {fig_path}")
        plt.close(fig)

    # ----- main RAA diagnostic plot -----
    output_path = run_dir / "raa_diagnostic.png"

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

    print(results[0].keys())