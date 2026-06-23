"""
schema.py
---------
Dataclass definitions for all experiment configuration.

Serializable fields only — no callables, no nn.Module instances.
History factories and log fns are resolved at runtime via core.pinn.registry
from cfg.param_model_type.

Usage
-----
    from experiment.configs.presets.simple import default_simple_config
    from core.pinn.factory import build_param_model
    from core.pinn.trainer import build_and_train_pinn

    cfg = default_simple_config()
    ctx = ParamModelContext()   # empty for constant PINN
    param_model = build_param_model(cfg, ctx)
    result = build_and_train_pinn(t_np, C_np, cfg, param_model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Supported values — used by registry and factory (not enforced at runtime yet)
PARAM_MODEL_TYPES = ("constant", "segment", "sigmoid_cp", "multi_sigmoid_cp")
COLLOCATION_TYPES = ("uniform", "piecewise")


@dataclass
class PhysicsConfig:
    """Known physical constants for the IAQ ODE."""
    V: float = 100.0
    C_out: float = 420.0
    C0: float = 500.0


@dataclass
class TrainConfig:
    """Hyperparameters shared by all PINN training variants."""
    n_hidden: int = 3
    hidden_dim: int = 64
    n_colloc: int = 1000
    lr_net: float = 3e-3
    lr_params: float = 1e-2
    min_lr_net: float = 1e-5
    min_lr_params: float = 1e-3
    epochs: int = 8000
    warmup_epochs: int = 500
    lambda_phys: float = 1.0
    ramp_epochs: int = 2000
    log_Q_init: float = field(default_factory=lambda: float(np.log(1.0)))
    log_S_init: float = field(default_factory=lambda: float(np.log(1.0)))
    kappa: float = 50.0
    seed: int = 42
    boundary_offset: float = 0.02   # piecewise collocation only


@dataclass
class Stage1Config:
    """RAA Stage I sliding-window scan parameters (no neural network)."""
    window_size: int = 20
    sigma: float = 1.5
    prominence: float | None = None
    prominence_factor: float = 0.15
    distance: int = 20
    margin_h: float = 0.4
    k: int | None = None

    
@dataclass
class DataConfig:
    x_col: str = "t"
    y_col: str = "C_meas"
    extra_cols: list[str] = field(default_factory=list)
    test_size: float = 0.2
    seed: int = 42


@dataclass
class ParamModelContext:
    """
    Runtime values needed to construct a param model — not known at preset time.

    Pipelines fill this after loading data / running Stage I, then pass to
    build_param_model(cfg, ctx).

    Fields used per param_model_type:
        constant          — (none)
        segment           — (n_segments, segment_duration from cfg)
        sigmoid_cp        — t_left, t_right
        multi_sigmoid_cp  — t_min, t_max, tau_inits
    """
    t_left: float | None = None
    t_right: float | None = None
    t_min: float | None = None
    t_max: float | None = None
    tau_inits: list[float] | None = None


@dataclass
class ExperimentConfig:
    """
    Top-level config for one experiment run.

    Declarative, JSON-serializable choices. Wiring (history_factory, log_fn)
    is resolved from param_model_type via core.pinn.registry.
    """
    name: str = "simple_pinn"
    dataset_path: str = ""
    param_model_type: str = "constant"
    collocation_type: str = "uniform"
    run_dir: Path | None = None
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)

    # RAA
    stage2_mode: str = "per_interval"   # "per_interval" | "one_pinn"

    # Varying PINN
    n_segments: int | None = None
    segment_duration: float | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """Load ExperimentConfig from YAML. TODO: implement with PyYAML or OmegaConf."""
    raise NotImplementedError(
        f"YAML loading not implemented yet. Use presets or build ExperimentConfig manually. ({path})"
    )
