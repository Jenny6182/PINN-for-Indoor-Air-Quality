## Experiment Results

This directory contains saved outputs from hyperparameter tuning, validation, ensemble experiments, and other completed experiments.

### `stage1_cache.json`

Contains precomputed Stage I RAA-PINN results for the validation datasets, including detected changepoint initializations and time-domain information.

The cache is used during Stage II hyperparameter tuning so that Stage I does not need to be rerun for every trial, reducing the computational cost of the optimization process.

### `summary1st_tuning.csv`

Contains the results of the first Optuna hyperparameter tuning round for the RAA-PINN.

The initial search used a relatively limited hyperparameter range. Several of the best-performing parameters were found near the boundaries of their search ranges, suggesting that the optimal values might lie outside the ranges considered. A second tuning round was therefore performed with substantially expanded search ranges.

### `summary2nd_tuning.csv`

Contains the results of the second Optuna hyperparameter tuning round for the RAA-PINN.

Although the search ranges were significantly expanded, some of the best-performing parameters remained near the boundaries. Rather than continuing to repeatedly expand the search space, the tuning process was stopped at this point, as further expansion was not expected to produce substantial improvements. The selected values from this tuning round were incorporated into the default RAA-PINN configuration preset.

### Other Result Subdirectories

The remaining subdirectories contain saved results and outputs from completed experiments, including hyperparameter sweeps, Optuna studies, validation runs, ensemble experiments, result-analysis plots, constsant pinn results, varying pinn results, etc.