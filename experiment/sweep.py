"""
sweep.py
--------
Grid search over hyperparameters (replaces inline loops in batch_run.py).

Usage (once implemented):
    from experiment.sweep import grid_sweep
    from experiment.configs.presets.raa import default_raa_config

    grid_sweep(
        base_cfg=default_raa_config(),
        grid={"train.log_Q_init": [np.log(50), np.log(100)], "train.log_S_init": [...]},
        out_dir="results/sensitivity_analysis",
    )

TODO (you implement):
    - itertools.product over grid keys
    - for each combo: clone base_cfg, apply overrides, call run_experiment(cfg)
    - append rows to master_summary.csv (reuse Tee pattern from batch_run.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment.configs.schema import ExperimentConfig


def grid_sweep(
    base_cfg: ExperimentConfig,
    grid: dict[str, list[Any]],
    out_dir: str | Path = "results/sweeps",
) -> list[dict[str, Any]]:
    """
    Run run_experiment() for every combination in grid.

    Parameters
    ----------
    base_cfg
        Starting config; grid values override nested fields by dotted key
        (e.g. "train.epochs", "stage1.prominence_factor").
    grid
        {dotted_field_path: [value1, value2, ...], ...}
    out_dir
        Root directory; each run gets a subdirectory.

    Returns
    -------
    list of result dicts, one per run.
    """
    raise NotImplementedError
