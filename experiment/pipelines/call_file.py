from experiment.configs.presets.raa import default_raa_config
from experiment.configs.schema import TrueValues
import pandas as pd
from experiment.pipelines.one_raapinn import raa_pipeline


if __name__ == "__main__":
    print("HI!")
    cfg = default_raa_config(run_dir="results/run1", dataset_path="data/datasets/varying_pinn_datasets/varying_Q.csv")
    print("got config successfully")

    print("Now printing config...")
    print(cfg)

    print("Now building true values...")
    true_vals = TrueValues(
        Q=pd.read_csv(cfg.data.dataset_path)["Q_true"].tolist(),
        S=pd.read_csv(cfg.data.dataset_path)["S_true"].tolist(),
    )

    print(true_vals)

    print("Now running raa pipelines...")
    raa_pipeline(cfg, true_vals=true_vals)