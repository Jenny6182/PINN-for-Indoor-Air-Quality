"""
Default configuration for the varying (piecewise Q, S) PINN pipeline.
"""

from experiment.configs.schema import ExperimentConfig, TrainConfig


def default_varying_config(
    dataset_path: str = "./data/datasets/varying_pinn_datasets/varying_Q.csv",
    n_segments: int = 3,
    segment_duration: float = 2.0,
) -> ExperimentConfig:
    """Return an ExperimentConfig ready for varying_pinn.run(cfg)."""
    return ExperimentConfig(
        name="varying_pinn",
        dataset_path=dataset_path,
        param_model_type="segment",
        collocation_type="piecewise",
        n_segments=n_segments,
        segment_duration=segment_duration,
        train=TrainConfig(),
    )
