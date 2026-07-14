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
PARAM_MODEL_TYPES = ("constant", "segment", "multi_sigmoid_cp")


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
    n_colloc: int = 800
    lr_net: float = 0.0039314657977675685
    lr_params: float = 1.1229029692286944e-05
    min_lr_net: float = 1e-5
    min_lr_params: float = 1e-3
    epochs: int = 8000
    warmup_epochs: int = 1500
    lambda_phys: float = 9.32102526150741
    ramp_epochs: int = 1100
    log_Q_init: float = field(default_factory=lambda: float(np.log(200.0)))
    log_S_init: float = field(default_factory=lambda: float(np.log(1e5)))
    kappa: float = 75.03986940162258
    seed: int = 42
    boundary_offset: float = 0.02   # piecewise collocation only
    print_every: int = 500
    verbose: bool = True

@dataclass
class Stage1Config:
    """RAA Stage I sliding-window scan parameters (no neural network)."""
    window_size: int = 25
    sigma: float = 6.177938360791919
    prominence: float | None = None
    prominence_factor: float = 0.007477744737290231
    distance: int = 24
    margin_h: float = 0.5695784403542213
    k: int | None = None

    
@dataclass
class DataConfig:
    dataset_path: str = "" # path to csv
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
        multi_sigmoid_cp  — t_min, t_max, tau_inits
    """
    t_min: float | None = None
    t_max: float | None = None
    tau_inits: list[float] | None = None

@dataclass
class TrueValues:
    Q: list[float] | None = None
    S: list[float] | None = None


@dataclass
class ExperimentConfig:
    """
    Top-level config for one experiment run.

    Declarative, JSON-serializable choices. Wiring (history_factory, log_fn)
    is resolved from param_model_type via core.pinn.registry.
    """
    # -- required --
    name: str = "simple_pinn" # used for labels output
    param_model_type: str = "constant" # constant | segment | multi_sigmoid_cp
    run_dir: Path | None = None # where to save outputs/plots

    physics: PhysicsConfig = field(default_factory=PhysicsConfig) # physics constants
    train: TrainConfig = field(default_factory=TrainConfig) # trainer hyperparams
    data: DataConfig = field(default_factory=DataConfig) # where's the csv, which columns to extract in the csv

    # -- optional --
    # raa
    stage1: Stage1Config = field(default_factory=Stage1Config) # for raa

    # Varying PINN
    n_segments: int | None = None # for factory and collocation
    segment_duration: float | None = None # for factory and collocation
