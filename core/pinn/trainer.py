"""
trainer.py
----------
Shared PINN training recipe used by every pipeline variant.

Call pattern
------------
    from core.pinn.factory import build_param_model
    from core.pinn.trainer import build_and_train_pinn
    from experiment.configs.schema import ParamModelContext

    ctx = ParamModelContext(t_left=..., t_right=...)   # if RAA per-interval
    param_model = build_param_model(cfg, ctx)
    result = build_and_train_pinn(t_np, C_np, cfg, param_model)

cfg supplies hyperparams (cfg.train, cfg.physics) and tags (param_model_type,
collocation_type). Registry resolves history_factory / log_fn from param_model_type.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from core.pinn.collocation import create_piecewise_collocation, create_uniform_collocation
from core.pinn.registry import get_variant
from core.utils.logger import make_printing_log_fn, print_header
from experiment.configs.schema import ExperimentConfig, ParamModelContext
from core.pinn.pinn_architecture import ParamModel


def _build_collocation_grid(t_norm: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Return collocation points in normalized time from cfg.collocation_type."""
    train = cfg.train
    if cfg.collocation_type == "uniform":
        return create_uniform_collocation(train.n_colloc, t_norm)
    if cfg.collocation_type == "piecewise":
        if cfg.segment_duration is None:
            raise ValueError("piecewise collocation requires cfg.segment_duration")
        return create_piecewise_collocation(
            train.n_colloc,
            t_norm,
            cfg.segment_duration,
            train.boundary_offset,
        )
    raise ValueError(f"Unknown collocation_type: {cfg.collocation_type!r}")


def build_and_train_pinn(
    t_np: np.ndarray,
    C_np: np.ndarray,
    cfg: ExperimentConfig,
    param_model: ParamModel,
    *,
    print_every: int = 500,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Assemble and train one PINN on the given (t, C) slice.

    Parameters
    ----------
    t_np, C_np:
        Time and CO₂ arrays, shape (N, 1). Sliced to the training window by the caller.
    cfg:
        Full experiment config — uses cfg.train, cfg.physics, cfg.param_model_type,
        cfg.collocation_type. History/log wiring comes from registry via param_model_type.
    print_every:
        Terminal progress row interval.
    verbose:
        Print training header and rows.

    Returns
    -------
    dict with keys:
        model, history, stats, t_np, C_np, t_col_np, t_norm, C_norm

    """
    variant = get_variant(cfg.param_model_type)

    if verbose:
        print_header(pinn_type=variant.pinn_type)

    # data preprocessing
    preprocessed_data = preprocess_data(t_np, C_np)

    # create collocation points 
    t_col_np = _build_collocation_grid(preprocessed_data["t_norm"], cfg)
    T_col    = to_torch(t_col_np, requires_grad=True)

     # build model
    net = FeedForwardNet(hidden_dim=cfg.train.hidden_dim, n_hidden=cfg.train.n_hidden)
    model = PINN(net, param_model)

    # optimizers and schedulers
    net_params = list(model.net.parameters())
    phys_params = list(model.param_model.parameters())

    opt_net = torch.optim.Adam(net_params,  lr=cfg.train.lr_net)
    opt_params = torch.optim.Adam(phys_params, lr=cfg.train.lr_params)

    sched_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_net, T_max=cfg.train.epochs, eta_min=cfg.train.min_lr_net
    )
    sched_params = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_params, T_max=cfg.train.epochs, eta_min=cfg.train.min_lr_params
    )

    # history and training
    history = variant.history_factory()

    train_loop(model, opt_net, opt_params, sched_net, sched_params,
               preprocessed_data["T_train"], preprocessed_data["C_train"], T_col, 
               preprocessed_data["stats"],
               cfg.train.epochs, cfg.train.warmup_epochs, cfg.train.lambda_phys, cfg.train.ramp_epochs,
               history,
               physics_kwargs={"V": cfg.physics.V, "C_out": cfg.physics.C_out},
               log_fn=make_printing_log_fn(variant.log_fn, variant.pinn_type, print_every, verbose))

    return {
    "model": model,
    "history": history,
    "stats": preprocessed_data[stats],
    "t_np": t_np,  
    "C_np": C_np,
    "t_col_np": t_col_np,
    "t_norm": preprocessed_data["t_norm"],
    "C_norm": preprocessed_data["C_norm"],
}