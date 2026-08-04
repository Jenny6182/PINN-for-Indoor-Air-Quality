from experiment.configs.presets.const_ra import constrained_raa_config
from experiment.pipelines.steady_state_pipeline import steady_state_pipeline


if __name__ == "__main__":
    print("Algorithm Started")
    cfg = constrained_raa_config(run_dir="results/run3", dataset_path="data/datasets/batch_run_simple_datasets/iaq_Q100_S0.30.csv")
    print("Obtained config successfully")

    print("Now running steady state pipeline...")
    steady_state_pipeline(cfg)