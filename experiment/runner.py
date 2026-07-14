"""
runner.py
---------
Experiment orchestration: one entry point for every pipeline.

Responsibilities (you implement incrementally):
  - Create a timestamped run directory under results/
  - Save cfg as config.json for reproducibility
  - Tee stdout to run.log
  - Dispatch to the correct pipeline via PIPELINE_REGISTRY
  - Save history.json and trigger loss/diagnostic plots after training

Pipelines stay thin — they only contain domain logic.
This module handles everything about *running* an experiment.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from experiment.configs.schema import ExperimentConfig

# Populated once thin pipelines exist. Keys match ExperimentConfig.name.
PIPELINE_REGISTRY: dict[str, Callable[[ExperimentConfig], dict[str, Any]]] = {
    # "simple_pinn": simple_pinn.run,
    # "varying_pinn": varying_pinn.run,
    # "raa_pinn": raa_pinn.run,
}


class RunContext:
    """
    Manages artifacts for a single experiment run.

    TODO (you implement):
        - __init__: create run_dir, save config.json
        - tee_stdout(): context manager redirecting print → run.log
        - save_history(history): write history.json
        - save_summary(metrics): write summary.csv
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir or self._default_run_dir(cfg)
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # TODO: self._save_config()

    @staticmethod
    def _default_run_dir(cfg: ExperimentConfig) -> Path:
        stem = Path(cfg.dataset_path).stem if cfg.dataset_path else cfg.name
        return Path(f"results/{cfg.name}/{stem}")

    def save_config(self) -> None:
        """Write cfg to run_dir/config.json."""
        path = self.run_dir / "config.json"
        with open(path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=2, default=str)

    def save_history(self, history: dict) -> None:
        """Write per-epoch training history to run_dir/history.json."""
        raise NotImplementedError

    def save_summary(self, metrics: dict) -> None:
        """Write scalar results to run_dir/summary.csv."""
        raise NotImplementedError


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Run one full experiment end-to-end.

    Parameters
    ----------
    cfg
        Complete experiment configuration. Use a preset or build manually.

    Returns
    -------
    dict
        Pipeline-specific results (metrics, history, stage2_results, ...).

    TODO (you implement):
        1. ctx = RunContext(cfg); cfg.run_dir = ctx.run_dir
        2. set torch/numpy seed from cfg.train.seed
        3. pipeline = PIPELINE_REGISTRY[cfg.name]
        4. with ctx.tee_stdout(): result = pipeline(cfg)
        5. ctx.save_history(result["history"]); ctx.save_summary(...)
        6. return result
    """
    if cfg.name not in PIPELINE_REGISTRY:
        raise KeyError(
            f"No pipeline registered for name={cfg.name!r}. "
            f"Available: {list(PIPELINE_REGISTRY)}. "
            f"Register your pipeline in PIPELINE_REGISTRY once run() is implemented."
        )
    raise NotImplementedError("run_experiment is not wired up yet.")
