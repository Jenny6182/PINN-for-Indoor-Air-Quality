"""
Default configuration for the simple (constant Q, S) PINN pipeline.
"""

from experiment.configs.schema import ExperimentConfig, TrainConfig


def default_simple_config(
    dataset_path: str = "./data/datasets/batch_run_simple_datasets/iaq_co2_simple.csv",
) -> ExperimentConfig:
    """Return an ExperimentConfig ready for simple_pinn.run(cfg)."""
    return ExperimentConfig(
        name="simple_pinn",
        dataset_path=dataset_path,
        param_model_type="constant",
        collocation_type="uniform",
        train=TrainConfig(),
    )
