

import numpy as np
import torch
from core.utils.preprocessing import normalize

def create_uniform_collocation(n_colloc, x):
    """Return n_colloc number of uniformly distributed points between x.min() and x.max()"""
    return np.linspace(x.min(), x.max(), n_colloc).reshape(-1, 1).astype(np.float32) # inclusive
    # get n_colloc number of evenely spaced points between x.min() and x.max()
    # return np array, and reshaped into (number of points, 1), with type float


def to_torch(x, requires_grad=False):
    """Given numpy array x, return tensor X"""
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)


def create_piecewise_collocation(n_colloc, x, segment_duration, boundary_offset):
    """
    Uniform grid plus extra points around each segment boundary,
    so the physics loss will be forced to evaluate and capture the behaviour at each Q/S jump better
    """
    x_uniform = create_uniform_collocation(n_colloc, x).flatten() # flatten the (n_colloc, 1) into (n_colloc, )
    
    # get all the boundary points so we can add points near them
    # create values starting at segment_duration, increasing by segment_duration every step, stopping before x.max()
    boundaries = np.arange(segment_duration, x.max(), segment_duration)

    # adding points before and after each boundary so residual will be evaluated around the discontinuity
    # joining them into one array
    x_near = np.concatenate([boundaries - boundary_offset, boundaries + boundary_offset])

    # combines uniform collocation with boundary collocation points
    # with duplicates removed and sorted in increasing order
    x_col = np.sort(np.unique(np.concatenate([x_uniform, x_near])))

    return x_col.reshape(-1, 1).astype(np.float32)
