import pandas as pd
import numpy as np

# this is for plotting and analysis purpose

def find_true_segments(
    csv_path,
    q_col="Q_true",
    s_col="S_true",
    tolerance=1e-8,
):
    """
    Extract true parameter change segments from CSV.

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