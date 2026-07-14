import time
from stage2_optuna import run_one, PRELOADED, SUBSET_NAMES
from experiment.configs.schema import TrainConfig
import numpy as np

name = SUBSET_NAMES[0]
train_cfg = TrainConfig(epochs=3000, n_hidden=4, hidden_dim=64, verbose=True, print_every=500,
                        log_Q_init=float(np.log(200.0)),
                        log_S_init=float(np.log(1e5)),)

start = time.time()
recon, cp = run_one(name, train_cfg)
print(f"One run took {time.time() - start:.1f} seconds")
print(recon, cp)