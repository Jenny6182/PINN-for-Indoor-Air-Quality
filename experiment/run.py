"""
run.py
------
CLI entry point for single experiment runs.

Usage (once implemented):
    python -m experiment.run --preset simple
    python -m experiment.run --preset raa --dataset path/to.csv --train.epochs 5000

TODO (you implement):
    argparse with --preset {simple,varying,raa}
    optional overrides for nested fields (--train.epochs, --stage1.prominence_factor, ...)
    calls run_experiment(cfg)
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "CLI not implemented yet. For now, call pipeline main() directly or use presets:\n"
        "  from experiment.configs.presets.simple import default_simple_config\n"
        "  from experiment.runner import run_experiment\n"
        "  run_experiment(default_simple_config())"
    )


if __name__ == "__main__":
    main()
