

import numpy as np
import torch
from Preprocessing import normalize

def create_uniform_collocation(n_colloc, x):
    return np.linspace(x.min(), x.max(), n_colloc).reshape(-1, 1).astype(np.float32)


def to_torch(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def create_piecewise_collocation(n_colloc, x, segment_duration, boundary_offset):
    """
    Uniform grid plus extra points around each segment boundary,
    so the physics loss will be forced to evaluate and capture the behaviour at each Q/S jump better
    """
    x_uniform = create_uniform_collocation(n_colloc, x).flatten()
    
    # create values starting at segment_duration, increasing by segment_duration, stopping before x.max()
    boundaries = np.arange(segment_duration, x.max(), segment_duration)

    # adding points near boundary (before and after) so residual will be e valuated around the discontinuity
    # joining them into one array
    x_near = np.concatenate([boundaries - boundary_offset, boundaries + boundary_offset])

    # combines uniform collocation with boundary collocation points
    # with duplicates removed and sorted in increasing order
    x_col = np.sort(np.unique(np.concatenate([x_uniform, x_near])))

    return x_col.reshape(-1, 1).astype(np.float32)
