# Usage: python -m steady_state_algorithm.known_segments.pinn.py


import pandas as pd
import numpy as np

from experiment.pipelines.constant import run as constant_pipeline
from core.utils.truth import find_true_segments
from core.utils.plotting import *
from experiment.configs.schema import *


import numpy as np
import pandas as pd


def make_segment_summary(
    results,
    t,
    C,
    Q_true,
    S_true,
    segments,
    C_out,
):
    rows = []

    for result, (start, end) in zip(results, segments):

        i = result["segment_idx"]

        C_seg = C[start:end]
        t_seg = t[start:end]

        # True constant parameters for this segment
        Q_true_seg = Q_true[start]
        S_true_seg = S_true[start]

        # PINN estimates
        Q_pred = float(np.asarray(result["estimates"]["Q"]).squeeze())
        S_pred = float(np.asarray(result["estimates"]["S"]).squeeze())

        # True steady-state concentration
        Css_true = C_out + S_true_seg / Q_true_seg

        rows.append({
            "segment": i,

            "start_idx": start,
            "end_idx": end,
            "n_points": len(C_seg),
            "duration_hr": t_seg[-1] - t_seg[0],

            # CO2 characteristics
            "C0": C_seg[0],
            "C_end": C_seg[-1],
            "delta_C": C_seg[-1] - C_seg[0],
            "abs_delta_C": abs(C_seg[-1] - C_seg[0]),
            "C_mean": C_seg.mean(),
            "C_std": C_seg.std(),
            "C_range": C_seg.max() - C_seg.min(),

            # True parameters
            "Q_true": Q_true_seg,
            "S_true": S_true_seg,
            "Css_true": Css_true,

            # PINN estimates
            "Q_pred": Q_pred,
            "S_pred": S_pred,

            # Parameter errors
            "Q_abs_error": abs(Q_pred - Q_true_seg),
            "S_abs_error": abs(S_pred - S_true_seg),

            "Q_rel_error": (
                abs(Q_pred - Q_true_seg) / abs(Q_true_seg)
            ),
            "S_rel_error": (
                abs(S_pred - S_true_seg) / abs(S_true_seg)
            ),

            # Distance from steady state
            "dist_start_to_Css": abs(C_seg[0] - Css_true),
            "dist_end_to_Css": abs(C_seg[-1] - Css_true),

            # PINN preprocessing stats
            "pinn_t_min": result["stats"]["t_min"],
            "pinn_t_max": result["stats"]["t_max"],
            "pinn_y_mean": result["stats"]["y_mean"],
            "pinn_y_std": result["stats"]["y_std"],
        })

    return pd.DataFrame(rows)


def run(
    csv_path: str,
    cfg,
    t_col="t_hours",
    conc_col="C_meas_ppm",
    q_col="Q_true",
    s_col="S_true",
):
    """
    Run constant PINN on every true parameter segment and plot results.

    Returns:
        dict:
            results: list of PINN result dictionaries
            segments: list of (start_idx, end_idx)
            data: dataframe
    """

    # Load full dataset
    df = pd.read_csv(csv_path)

    t = df[t_col].to_numpy()
    C = df[conc_col].to_numpy()

    Q_true = df[q_col].to_numpy()
    S_true = df[s_col].to_numpy()

    # Extract true segment indices
    segments, changepoints = find_true_segments(
        csv_path,
        q_col=q_col,
        s_col=s_col,
    )

    for start,end in segments:
        print(start,end,t[start],t[end-1])

    plot_extracted_segments(
        t,
        C,
        segments,
        output_path="segments_debug.png",
    )

    results = []

    # Train one constant PINN per segment
    for i, (start, end) in enumerate(segments):

        print(f"\nTraining segment {i}: [{start}, {end})")

        t_seg = t[start:end].reshape(-1, 1)
        C_seg = C[start:end].reshape(-1, 1)

        c_mean = C_seg.mean()
        c_std = C_seg.std()

        result = constant_pipeline(
            t_seg,
            C_seg,
            cfg,
        )

        result["segment_idx"] = i
        result["start_idx"] = start
        result["end_idx"] = end

        results.append(result)

    summary_df = make_segment_summary(
        results,
        t,
        C,
        Q_true,
        S_true,
        segments,
        cfg.physics.C_out,
    )

    print(summary_df)
    summary_df.to_csv("segment_summary.csv", index=False)

    # Plot all segment results
    plot_all_segments(
        results=results,
        t=t,
        C=C,
        Q_true=Q_true,
        S_true=S_true,
        segments=segments,
        cfg=cfg,
        output_path="segments.png",
    )

    return {
        "results": results,
        "segments": segments,
        "changepoints": changepoints,
        "data": df,
    }



def main():
    csv_path = (
        "data/datasets/no_noise_dataset_S/iaq_varyS_seg11_seed87.csv"
    )

    # Uses all defaults from schema.py
    cfg = ExperimentConfig()

    # Ensure this pipeline uses constant Q/S parameters
    cfg.param_model_type = "constant"

    results = run(
        csv_path,
        cfg,
    )

    print("\nFinished training.")
    print(f"Number of segments: {len(results['results'])}")

    for i, result in enumerate(results["results"]):
        estimates = result["estimates"]

        print(
            f"Segment {i}: "
            f"Q={estimates['Q'][0]:.4f}, "
            f"S={estimates['S'][0]:.4f}"
        )


if __name__ == "__main__":
    main()


# This script feeds true segment changepoint into pinn to train on per segment.
# See "steady_state_algorithm_I_pinn_II_result" for results of this.
# Idea is that if it doesn't perform well even with true stage 1, it wouldn't work well with inaccurate
# stage1 data derived from the algorithmic method.