"""
Default configurations for RAA-PINN pipelines.
"""

import numpy as np

from experiment.configs.schema import DataConfig, ExperimentConfig, Stage1Config, TrainConfig


_STAGE2_PARAM_TYPE = {
    "per_interval": "sigmoid_cp",
    "one_pinn": "multi_sigmoid_cp",
}

def default_raa_config(
    dataset_path: str = "./data/datasets/varying_pinn_datasets/varying_Q.csv",
    stage2_mode: str = "per_interval",
    log_Q_init: float | None = None,
    log_S_init: float | None = None,
) -> ExperimentConfig:
    """
    Return an ExperimentConfig ready for raa_pinn.run(cfg).

    param_model_type is set automatically from stage2_mode:
        per_interval → sigmoid_cp
        one_pinn     → multi_sigmoid_cp
    """
    if stage2_mode not in _STAGE2_PARAM_TYPE:
        raise ValueError(
            f"stage2_mode={stage2_mode!r} must be one of {list(_STAGE2_PARAM_TYPE)}"
        )
    return ExperimentConfig(
        name="raa_pinn",
        dataset_path=dataset_path,
        stage2_mode=stage2_mode,
        param_model_type=_STAGE2_PARAM_TYPE[stage2_mode],
        collocation_type="uniform",
        stage1=Stage1Config(),
        data=DataConfig(                       
            x_col="t_hours",
            y_col="C_meas_ppm",
            extra_cols=["Q_true", "S_true"],
        ),
        train=TrainConfig(
            n_hidden=2,
            hidden_dim=32,
            n_colloc=1000,
            epochs=3000,
            warmup_epochs=200,
            ramp_epochs=500,
            log_Q_init=log_Q_init if log_Q_init is not None else float(np.log(200.0)),
            log_S_init=log_S_init if log_S_init is not None else float(np.log(1e5)),
        ),
    )

