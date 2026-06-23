"""
registry.py
-----------
Maps cfg.param_model_type → logging callbacks and terminal labels.

Keep callables here, NOT in ExperimentConfig (so config stays serializable).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from core.utils.logger import (
    log_fn_one_pinn,
    log_fn_simple,
    log_fn_stage2,
    log_fn_varying,
    make_history_one_pinn,
    make_history_simple,
    make_history_stage2,
    make_history_varying,
)
from experiment.configs.schema import COLLOCATION_TYPES, PARAM_MODEL_TYPES


@dataclass(frozen=True)
class PinnVariant:
    """Logging and history wiring for one param_model_type."""
    history_factory: Callable[[], dict]
    log_fn: Callable[[nn.Module, dict, int], None]
    pinn_type: str   # label for print_header / print_row


VARIANT_REGISTRY: dict[str, PinnVariant] = {
    "constant": PinnVariant(
        history_factory=make_history_simple,
        log_fn=log_fn_simple,
        pinn_type="simple",
    ),
    "segment": PinnVariant(
        history_factory=make_history_varying,
        log_fn=log_fn_varying,
        pinn_type="varying",
    ),
    "sigmoid_cp": PinnVariant(
        history_factory=make_history_stage2,
        log_fn=log_fn_stage2,
        pinn_type="stage2",
    ),
    "multi_sigmoid_cp": PinnVariant(
        history_factory=make_history_one_pinn,
        log_fn=log_fn_one_pinn,
        pinn_type="one_pinn",
    ),
}


def get_variant(param_model_type: str) -> PinnVariant:
    """Return registry entry for param_model_type; raise KeyError if unknown."""
    if param_model_type not in VARIANT_REGISTRY:
        raise KeyError(
            f"Unknown param_model_type={param_model_type!r}. "
            f"Choose one of: {list(VARIANT_REGISTRY)}"
        )
    return VARIANT_REGISTRY[param_model_type]


def validate_config_tags(param_model_type: str, collocation_type: str) -> None:
    """Raise ValueError for unsupported tag strings (optional sanity check)."""
    if param_model_type not in PARAM_MODEL_TYPES:
        raise ValueError(
            f"param_model_type={param_model_type!r} not in {PARAM_MODEL_TYPES}"
        )
    if collocation_type not in COLLOCATION_TYPES:
        raise ValueError(
            f"collocation_type={collocation_type!r} not in {COLLOCATION_TYPES}"
        )
