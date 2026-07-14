from experiment.configs.presets.raa import default_raa_config
from experiment.configs.schema import TrueValues
import pandas as pd
from experiment.pipelines.one_raapinn import raa_pipeline


if __name__ == "__main__":
    print("Training Started")
    cfg = default_raa_config(run_dir="results/run1", dataset_path="data/datasets/validation_dataset/iaq_varyQ_seg5_seed57.csv")
    print("Obtained config successfully")

    print("Now printing config...")

    print("Now building true values...")
    true_vals = TrueValues(
        Q=pd.read_csv(cfg.data.dataset_path)["Q_true"].tolist(),
        S=pd.read_csv(cfg.data.dataset_path)["S_true"].tolist(),
    )

    print("Now running raa pipelines...")
    raa_pipeline(cfg, true_vals=true_vals)