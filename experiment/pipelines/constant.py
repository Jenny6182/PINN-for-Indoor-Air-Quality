# experiment/pipelines/constant.py

from __future__ import annotations

import numpy as np

from core.pinn.factory import build_param_model
from core.pinn.trainer import build_and_train_pinn
from experiment.configs.schema import ParamModelContext


def run(
    t_np: np.ndarray,
    C_np: np.ndarray,
    cfg,
):
    """
    Train one constant PINN on a single data segment.

    Parameters
    ----------
    t_np:
        Time array for one segment.
    C_np:
        CO2 concentration array for one segment.
    cfg:
        Experiment configuration.

    Returns
    -------
    result:
        Dictionary containing:
            model
            history
            estimates (Q, S)
            stats
            t_col_np
    """

    # constant parameter model has no segment context
    ctx = ParamModelContext()

    # build Q and S parameter model
    param_model = build_param_model(
        cfg,
        ctx,
    )

    # train PINN
    result = build_and_train_pinn(
        t_np,
        C_np,
        cfg,
        param_model,
    )

    return result