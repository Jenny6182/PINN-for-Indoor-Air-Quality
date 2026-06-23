import numpy as np

# NOTE: New code should use experiment.configs.schema (TrainConfig, PhysicsConfig).
# This module is kept for backward compatibility with existing pipelines.
# Defaults here may differ from schema.py — migrate imports when refactoring.

# known physical constants 
V = 100.0   # zone volume, m^3
C_out = 420.0   # outdoor CO2, ppm
C0 = 500.0   # initial CO2, ppm  (first data point)

# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
SEED = 42

# ------ hyper-parameters ------
N_HIDDEN = 3        # number of hidden layers
HIDDEN_DIM = 64       # number of neurons per layer
N_COLLOC = 1000     # number of collocation points
LR_NET = 3e-3     # learning rate for network weights
LR_PARAMS = 1e-2     # learning rate for log_Q, log_S parameters
EPOCHS = 8000     # total training epochs

# Total ramp up period is warmup+ramp epochs so 500+2000 = 2500 epochs until full weight on physics loss (balanced)
WARMUP_EPOCHS = 500      # how many epochs to use only data loss
LAMBDA_PHYS = 1.0      # final physics loss weight
RAMP_EPOCHS = 2000     # ramp lambda_phys over this many epochs after warmup

# initial guesses of Q and S
# True values: Q≈200, S≈1e5
# log so it's positive and similar scaled
LOG_Q_INIT = np.log(1) 
LOG_S_INIT = np.log(1)

# Hyperparams for stage I
WINDOW_SIZE  = 20      # points per sliding window (~20 min at 1-min sampling)
SIGMA        = 1.5     # Gaussian smoothing sigma before differentiation
PROMINENCE   = None    # set to None to auto-set as 15% of max score
# prominence_factor = 0.05 #default 0.15
DISTANCE     = 20      # min points between peaks (same as WINDOW_SIZE is safe)
MARGIN_H     = 0.4     # candidate interval half-width around each peak [hours]

N_HIDDEN_S2      = 2
HIDDEN_DIM_S2    = 32
N_COLLOC_S2      = 300
LR_NET_S2        = 3e-3
LR_PARAMS_S2     = 1e-2
EPOCHS_S2        = 3000
WARMUP_EPOCHS_S2 = 200
LAMBDA_PHYS_S2   = 1.0
RAMP_EPOCHS_S2   = 500
KAPPA            = 50.0    # sigmoid sharpness — higher = sharper step approximation