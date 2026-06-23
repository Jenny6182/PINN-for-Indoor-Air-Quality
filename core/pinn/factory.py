"""
factory.py
----------
Build param_model instances from ExperimentConfig + ParamModelContext.

Pipelines create ctx after loading data / Stage I, then call build_param_model.
The trainer receives the built module — it does not construct param models itself.
"""

from __future__ import annotations

import torch.nn as nn

from core.pinn.pinn_architecture import (
    ConstantParams,
    MultiSigmoidChangepoint,
    SegmentParams,
    SigmoidChangepoint,
)
from core.pinn.registry import validate_config_tags
from experiment.configs.schema import ExperimentConfig, ParamModelContext


def build_param_model(cfg: ExperimentConfig, ctx: ParamModelContext) -> ParamModel:
    """
    Construct the parameter model for cfg.param_model_type.

    Parameters
    ----------
    cfg
        ExperimentConfig with param_model_type and train init values.
    ctx
        Runtime bounds / changepoint guesses (see ParamModelContext docstring).

    Returns
    -------
    nn.Module
        One of ConstantParams, SegmentParams, SigmoidChangepoint,
        MultiSigmoidChangepoint.

  TODO (you implement):
        Copy constructors from stage2.py / one_stage2.py / pinn_architecture.py.
        Raise ValueError if required ctx fields are missing for the chosen type.
    """
    validate_config_tags(cfg.param_model_type, cfg.collocation_type)
    train = cfg.train

    if cfg.param_model_type == "constant":
        return ConstantParams(
            log_Q_init=train.log_Q_init,
            log_S_init=train.log_S_init,
        )

    if cfg.param_model_type == "segment":
        if cfg.n_segments is None or cfg.segment_duration is None:
            raise ValueError(
                "segment param model requires cfg.n_segments and cfg.segment_duration"
            )
        return SegmentParams(
            n_segments=cfg.n_segments,
            log_Q_init=train.log_Q_init,
            log_S_init=train.log_S_init,
            segment_duration=cfg.segment_duration,
        )

    if cfg.param_model_type == "sigmoid_cp":
        if ctx.t_left is None or ctx.t_right is None:
            raise ValueError("sigmoid_cp requires ctx.t_left and ctx.t_right")
        return SigmoidChangepoint(
            t_left=ctx.t_left,
            t_right=ctx.t_right,
            log_Q_init=train.log_Q_init,
            log_S_init=train.log_S_init,
            kappa=train.kappa,
        )

    if cfg.param_model_type == "multi_sigmoid_cp":
        if ctx.t_min is None or ctx.t_max is None or not ctx.tau_inits:
            raise ValueError(
                "multi_sigmoid_cp requires ctx.t_min, ctx.t_max, and ctx.tau_inits"
            )
        return MultiSigmoidChangepoint(
            t_min=ctx.t_min,
            t_max=ctx.t_max,
            tau_inits=ctx.tau_inits,
            log_Q_init=train.log_Q_init,
            log_S_init=train.log_S_init,
            kappa=train.kappa,
        )

    raise KeyError(f"Unhandled param_model_type: {cfg.param_model_type!r}")
