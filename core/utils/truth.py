import numpy as np

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