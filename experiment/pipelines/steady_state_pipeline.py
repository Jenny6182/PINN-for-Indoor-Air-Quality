"""
experiment/pipelines/steady_state_pipeline.py
-----------------------------------------------
Scenario 1: algorithmic steady/transient identification + direct algebraic
solve for Q/S, as an alternative to full PINN training (Stage II).

Steps:
    1. Load data
    2. Stage I - sliding window scan to detect changepoints (UNCHANGED,
       same as one_raapinn.py)
    3. NO refinement of taus -- Stage I's detected taus are used as-is to
       define segment boundaries. There is no Stage II training here, so
       there's nothing to jointly refine tau positions against.
    4. Within each segment, classify steady vs transient timepoints
    5. Solve directly (no training) for the per-segment varying parameter
       and the shared constant, using all steady points pooled across
       segments
    6. Compute the same reconstruction/changepoint metrics as the PINN
       pipeline, for direct comparison

This is intentionally NOT a ParamModel / not wired into core/pinn/factory.py
-- there's no training loop, no nn.Module, no gradients. It's a separate,
much cheaper method meant to be compared against the trained PINN's results
on the same datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from core.utils.truth import load_truth
from core.utils.evaluation import reconstruction_error, changepoint_metrics
from core.utils.preprocessing import prepare_training_data
from core.scan.window_sweeping import stage1_scan, find_candidate_intervals, find_top_k_intervals
from core.steady_state.classify import classify_steady_transient
from core.steady_state.solver import solve_joint_steady_state
from experiment.configs.schema import ExperimentConfig
from core.utils.plotting import plot_steady_state_results


def steady_state_pipeline(cfg: ExperimentConfig) -> dict:
    """
    Full steady-state (non-PINN) pipeline. Mirrors one_raapinn.py's
    structure/metrics/output format so results are directly comparable,
    but has no Stage II training.
    """

    run_dir = cfg.run_dir or Path(f"results/{cfg.name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Steady-State Pipeline: {cfg.name}")
    print("=" * 60)

    # --- load data ---
    data = prepare_training_data(
        path=cfg.data.dataset_path,
        x_col=cfg.data.x_col,
        y_col=cfg.data.y_col,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
    )

    truth = load_truth(cfg.data.dataset_path)
    truth_type = truth["type"] if truth is not None else None

    t_np = data["t_np"]
    C_meas_np = data["c_np"]
    t_flat = t_np.flatten()
    C_flat = C_meas_np.flatten()

    print(f"Loaded {len(t_flat)} timepoints "
          f"(t = {t_flat.min():.2f}h to {t_flat.max():.2f}h)")

    # --- stage 1 ---
    print("\n--- Stage I: Sliding Window Scan ---")

    scores = stage1_scan(
        t=t_np,
        C_meas=C_meas_np,
        V=cfg.physics.V,
        C_out=cfg.physics.C_out,
        window_size=cfg.stage1.window_size,
        sigma=cfg.stage1.sigma,
    )

    if cfg.stage1.k is not None:
        peak_indices, intervals = find_top_k_intervals(
            t=t_np,
            scores=scores,
            k=cfg.stage1.k,
            margin_h=cfg.stage1.margin_h,
        )
    else:
        prominence = (
            cfg.stage1.prominence
            or cfg.stage1.prominence_factor * np.nanmax(scores)
        )
        peak_indices, intervals = find_candidate_intervals(
            t=t_np,
            scores=scores,
            prominence=prominence,
            distance=cfg.stage1.distance,
            margin_h=cfg.stage1.margin_h,
        )

    tau_inits = sorted(float(t_flat[i]) for i in peak_indices)
    t_min = float(t_flat.min())
    t_max = float(t_flat.max())

    print(f"Detected {len(tau_inits)} candidate changepoints:")
    for i, tau in enumerate(tau_inits):
        print(f"  [{i+1}] tau = {tau:.3f}h")

    if len(tau_inits) == 0:
        print("No changepoints detected. Treating as one segment.")

    # --- build segments ---
    boundaries = [t_min] + tau_inits + [t_max]
    n_segments = len(boundaries) - 1

    segment_steady_data = []
    n_steady_per_segment = []

    print("\n--- Steady/Transient Classification per Segment ---")

    for i in range(n_segments):
        seg_lo, seg_hi = boundaries[i], boundaries[i + 1]

        if i == n_segments - 1:
            seg_mask = (t_flat >= seg_lo) & (t_flat <= seg_hi)
        else:
            seg_mask = (t_flat >= seg_lo) & (t_flat < seg_hi)

        t_seg = t_flat[seg_mask]
        C_seg = C_flat[seg_mask]

        if len(t_seg) < 3:
            print(
                f"  segment {i+1}: too few points ({len(t_seg)}), skipping"
            )
            segment_steady_data.append(
                {"C_steady": np.array([]), "dCdt_steady": np.array([])}
            )
            n_steady_per_segment.append(0)
            continue

        steady_mask = classify_steady_transient(
            t_seg,
            C_seg,
            sigma=cfg.stage1.sigma,
        )

        from scipy.ndimage import gaussian_filter1d

        C_smooth = gaussian_filter1d(C_seg, sigma=cfg.stage1.sigma)
        dCdt_seg = np.gradient(C_smooth, t_seg)

        C_steady = C_seg[steady_mask]
        dCdt_steady = dCdt_seg[steady_mask]

        n_steady = steady_mask.sum()
        n_steady_per_segment.append(int(n_steady))

        print(
            f"  segment {i+1}: "
            f"{n_steady}/{len(t_seg)} steady points"
        )

        segment_steady_data.append(
            {
                "C_steady": C_steady,
                "dCdt_steady": dCdt_steady,
            }
        )

    # --- solve ---
    print("\n--- Solving Steady-State Q/S ---")

    solution = solve_joint_steady_state(
        segment_steady_data,
        V=cfg.physics.V,
        C_out=cfg.physics.C_out,
        vary_param=cfg.train.vary_param,
    )

    varying_vals = solution["varying"]
    constant_val = solution["constant"]

    if cfg.train.vary_param == "Q":
        Q_seg = varying_vals
        S_seg = np.full(n_segments, constant_val)
    else:
        S_seg = varying_vals
        Q_seg = np.full(n_segments, constant_val)

    print(
        f"\nShared constant "
        f"({'S' if cfg.train.vary_param == 'Q' else 'Q'}): "
        f"{constant_val:.4g}"
    )

    # --- metrics ---
    taus_arr = np.array(tau_inits)

    if truth_type == "piecewise":
        pred_values = (
            Q_seg if cfg.train.vary_param == "Q"
            else S_seg
        )

        recon = reconstruction_error(
            truth["t"],
            truth["param_true"],
            taus_arr,
            pred_values,
            t_min,
            t_max,
        )

    elif truth_type == "constant":
        recon = {
            "Q_error": float(abs(Q_seg[0] - truth["Q_true"])),
            "S_error": float(abs(S_seg[0] - truth["S_true"])),
        }

    else:
        recon = {}

    if truth is not None:
        cp = changepoint_metrics(
            truth["true_taus"],
            taus_arr,
        )
    else:
        cp = {}

    metrics = {
        "dataset": str(cfg.data.dataset_path),
        "run_name": cfg.name,
        "truth_type": truth_type,
        "vary_param": cfg.train.vary_param,
        "method": "steady_state_algebraic",
        **{f"recon_{k}": v for k, v in recon.items()},
        **{f"cp_{k}": v for k, v in cp.items()},
        "stage1_sigma": cfg.stage1.sigma,
        "n_segments": n_segments,
        "n_steady_points_total": int(sum(n_steady_per_segment)),
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    master_path = Path("results/summary_steady_state.csv")
    master_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        master_path,
        mode="a",
        header=not master_path.exists(),
        index=False,
    )

    result = {
        "taus": taus_arr,
        "Q": Q_seg,
        "S": S_seg,
        "metrics": metrics,
        "n_steady_per_segment": n_steady_per_segment,
    }

    plot_steady_state_results(
        result={"Q": Q_seg, "S": S_seg, "taus": taus_arr},
        data=data,
        cfg=cfg,
        truth=truth,
        stage1_scores=scores,
        peaks=peak_indices,
        output_path=run_dir / "steady_state_results.png",
    )

    return result