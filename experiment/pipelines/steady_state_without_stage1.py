# experiments/run_steady_state.py

import numpy as np

from core.steady_state.classify import classify_steady_transient
from core.steady_state.solver import solve_joint_steady_state
from core.utils.plotting import plot_steady_state_results


def run_steady_state_pipeline(data, cfg, truth=None, output_path="steady_state.png"):
    t = data["t_np"].flatten()
    C = data["c_np"].flatten()

    # 1. classify steady/transient directly (no stage1)
    steady_mask = classify_steady_transient(t, C, sigma=1.5, eps_rel=0.02, min_duration_frac=0.15)

    if not np.any(steady_mask):
        raise ValueError("No steady-state region detected.")

    # 2. extract steady points
    t_steady = t[steady_mask]
    C_steady = C[steady_mask]

    # 3. estimate derivative only for solver compatibility
    dCdt_steady = np.gradient(C_steady, t_steady)

    segment_data = [{"C_steady": C_steady, "dCdt_steady": dCdt_steady}]

    # 4. solve Q and S from steady-state points
    # For now this assumes one segment
    Q_result = solve_joint_steady_state(segment_data, cfg.physics.V, cfg.physics.C_out, "Q")
    S_result = solve_joint_steady_state(segment_data, cfg.physics.V, cfg.physics.C_out, "S")

    result = {
        "Q": Q_result["varying"],
        "S": S_result["varying"],
        "taus": [],
    }

    # 5. plot classification + Q/S estimates
    plot_steady_state_results(
        result,
        data,
        cfg,
        steady_mask=steady_mask,
        truth=truth,
        output_path=output_path,
    )

    return result