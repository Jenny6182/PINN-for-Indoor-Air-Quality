"""
Default configurations for RAA-PINN pipelines.
"""

import numpy as np
from pathlib import Path
from experiment.configs.schema import DataConfig, ExperimentConfig, Stage1Config, TrainConfig

_STAGE2_PARAM_TYPE = {
    "per_interval": "sigmoid_cp",
    "one_pinn": "multi_sigmoid_cp",
    "one_pinn_constrained": "const_sigmoid_cp",
}

def constrained_raa_config(
    run_dir: str,
    dataset_path: str = "./data/datasets/varying_pinn_datasets/varying_Q.csv",
    vary_param: str = "Q",
    log_Q_init: float | None = None,
    log_S_init: float | None = None,
) -> ExperimentConfig:
    """Return ExperimentConfig for constrained (one-param-varies) RAA-PINN."""
    return ExperimentConfig(
        name="raa_pinn_constrained",
        run_dir=Path(run_dir),
        param_model_type="const_sigmoid_cp",
        data=DataConfig(
            dataset_path=dataset_path,
            x_col="t_hours",
            y_col="C_meas_ppm",
        ),
        train=TrainConfig(
            n_hidden=3,
            hidden_dim=64,
            n_colloc=800,
            lr_net=0.0039314657977675685,
            lr_params=1.1229029692286944e-05,
            epochs=8000,
            warmup_epochs=1500,
            ramp_epochs=1100,
            lambda_phys=50,  # tried to change from 9.32102526150741 to 50 to see if it improve physics constraint
            kappa=75.03986940162258,
            vary_param=vary_param,
            log_Q_init=log_Q_init if log_Q_init is not None else float(np.log(200.0)),
            log_S_init=log_S_init if log_S_init is not None else float(np.log(1e5)),
        ),
        stage1=Stage1Config(),
    )