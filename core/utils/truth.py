# truth.py

from pathlib import Path
import re
import numpy as np
import pandas as pd


def detect_truth_type(dataset_path, df):
    """
    Detects the type of truth available for the dataset.

    Returns
    -------
    str
        "piecewise" : Q_true/S_true columns exist and contain changes
        "constant"  : Q and S are constant values
        None        : no truth available
    """

    if "Q_true" in df.columns and "S_true" in df.columns:
        if df["Q_true"].nunique() > 1 or df["S_true"].nunique() > 1:
            return "piecewise"
        else:
            return "constant"

    filename = Path(dataset_path).stem
    if re.search(r"Q([0-9.]+)_S([0-9.]+)", filename):
        return "constant"

    return None


def load_truth(dataset_path):
    """
    Main truth loader.

    Returns
    -------
    truth : dict | None
    """

    df = pd.read_csv(dataset_path)

    truth_type = detect_truth_type(dataset_path, df)

    if truth_type == "piecewise":
        return load_truth_from_csv(df)

    elif truth_type == "constant":
        if "Q_true" in df.columns and "S_true" in df.columns:
            return load_constant_truth_from_columns(df)

        return load_constant_truth(dataset_path)

    return None


def load_truth_from_csv(df, tol=1e-9):
    """
    Loads truth from datasets containing Q_true/S_true columns.

    Extracts:
    - segment values
    - changepoint times
    """

    t = df["t_hours"].values

    Q_true = df["Q_true"].values
    S_true = df["S_true"].values

    Q_change_idx = np.where(np.abs(np.diff(Q_true)) > tol)[0] + 1
    S_change_idx = np.where(np.abs(np.diff(S_true)) > tol)[0] + 1

    change_idx = np.unique(
        np.concatenate((Q_change_idx, S_change_idx))
    )

    true_taus = t[change_idx]

    segments = []

    start = 0
    for idx in np.append(change_idx, len(t)):
        segments.append(
            {
                "Q": Q_true[start],
                "S": S_true[start],
            }
        )
        start = idx

    return {
        "type": "piecewise",
        "t": t,
        "segments": segments,
        "true_taus": true_taus,
        "Q_true": Q_true,
        "S_true": S_true,
    }


def load_constant_truth(dataset_path):
    """
    Loads constant Q/S truth from filename.

    Expected filename format:
        iaq_Q100_S0.30.csv
    """

    filename = Path(dataset_path).stem

    match = re.search(r"Q([0-9.]+)_S([0-9.]+)", filename)

    if match is None:
        return None

    Q = float(match.group(1))
    S = float(match.group(2))

    return {
        "type": "constant",
        "segments": [
            {
                "Q": Q,
                "S": S,
            }
        ],
        "true_taus": np.array([]),
        "Q_true": Q,
        "S_true": S,
    }


def load_constant_truth_from_columns(df):
    """
    Loads constant Q/S truth when Q_true/S_true columns exist
    but contain only one value.
    """

    Q = float(df["Q_true"].iloc[0])
    S = float(df["S_true"].iloc[0])

    return {
        "type": "constant",
        "segments": [
            {
                "Q": Q,
                "S": S,
            }
        ],
        "true_taus": np.array([]),
        "Q_true": Q,
        "S_true": S,
    }