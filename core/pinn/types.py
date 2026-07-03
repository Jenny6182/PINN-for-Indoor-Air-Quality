"""
types.py
--------
TypedDict definitions for data passed between core modules.

estimates   {"Q": np.ndarray, "S": np.ndarray, "taus": np.ndarray | None}
stats       {"t_min": float, "t_max": float, "y_mean": float, "y_std": float}
history     {"loss_total", "loss_data", "loss_phys", "Q", "S", "taus"}

"""

from __future__ import annotations
from typing import Any, TypedDict
import numpy as np


class PinnResult(TypedDict):
    model:      Any
    history:    dict
    stats:      dict
    t_col_np:   np.ndarray
    estimates:  dict        # {"Q": np.ndarray, "S": np.ndarray, "taus": np.ndarray | None}


class Stage1Result(TypedDict):
    t_np:         np.ndarray
    scores:       np.ndarray
    peak_indices: np.ndarray
    intervals:    list[tuple[float, float]]