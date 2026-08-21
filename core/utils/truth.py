import numpy as np
import pandas as pd

def extract_truth_from_dataframe(df, mode, tol=1e-9):
    """
    mode: 'Q' or 'S' — whichever one varies in this dataset.
    Returns true segment values + changepoint times, derived directly
    from the noise-free Q_true/S_true column.
    """
    t = df["t_hours"].values
    col = "Q_true" if mode == "Q" else "S_true"
    param_true = df[col].values

    change_idx = np.where(np.abs(np.diff(param_true)) > tol)[0] + 1
    true_taus  = t[change_idx]                          # changepoint times
    seg_values = np.concatenate(([param_true[0]], param_true[change_idx]))

    return {
        "t": t,
        "param_true": param_true,   # full array, for reconstruction error
        "true_taus": true_taus,     # for changepoint metrics
        "seg_values": seg_values,   # for TrueValues / printing
    }

def find_true_segments(
    csv_path,
    q_col="Q_true",
    s_col="S_true",
    tolerance=1e-8,
):
    """
    Extract true parameter-change segments from CSV.

    Returns:
        segments: list of (start_idx, end_idx)
        changepoints: indices where a new segment starts
    """

    df = pd.read_csv(csv_path)

    Q = df[q_col].to_numpy()
    S = df[s_col].to_numpy()

    # detect changes in either Q or S
    q_change = np.abs(np.diff(Q)) > tolerance
    s_change = np.abs(np.diff(S)) > tolerance

    change_points = np.where(q_change | s_change)[0] + 1

    n = len(df)

    boundaries = [0] + list(change_points) + [n]

    segments = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]

    return segments, change_points