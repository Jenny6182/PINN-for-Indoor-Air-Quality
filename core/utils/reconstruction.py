import numpy as np


def reconstruct_step_function(t_eval, taus, values):
    """
    Reconstruct a piecewise-constant trajectory from changepoints.

    The RAA-PINN model represents time-varying parameters (e.g., ventilation
    rate Q and source rate S) as piecewise-constant functions. This function
    converts the compact representation of changepoints and segment values
    into a dense trajectory evaluated at each time point in t_eval.

    Parameters
    ----------
    t_eval : np.ndarray
        Time points where the trajectory is evaluated.

    taus : array-like
        Estimated changepoint locations separating segments.

    values : array-like
        Estimated parameter value for each segment.

    Returns
    -------
    np.ndarray
        Reconstructed piecewise-constant trajectory over t_eval.
    """

    taus = np.asarray(taus).flatten()
    values = np.asarray(values).flatten()

    boundaries = [t_eval.min()] + list(taus) + [t_eval.max()]

    y = np.zeros_like(t_eval)

    for i in range(len(values)):
        mask = (t_eval >= boundaries[i]) & (t_eval < boundaries[i + 1])
        y[mask] = values[i]

    # include final time point
    y[-1] = values[-1]

    return y