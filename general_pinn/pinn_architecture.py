

# ------- imports --------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from config.config import V, C_out, C0, SEED


#------Set seed------
# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
torch.manual_seed(SEED) # set random numbers in torch
np.random.seed(SEED) # in numpy

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


#----- PINN class -----

# ----- set up PINN ------
# make a class that inherit from nn.Module
class FeedForwardNet(nn.Module):
    # define the layers and parameters and activation and stuff
    def __init__(self, hidden_dim=64, n_hidden=3):

        super().__init__()

        # layers use hidden_dim and n_hidden, pass in the hyperparams when using it, and activation is tanh()
        # this list will become arguments to nn.Sequential stacking them all together
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()] # make input layer, 1 input to hidden_dim neurons

        # make hidden layers
        for _ in range(n_hidden - 1):    # we are just counting n times, not using the variable _
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]

        # make output layer and add it in to layers list
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers) # sequentially stacking the layers, like the usual set up
        # use * to unpack the list into arguments

    # get normalized C prediction
    def forward(self, t_norm):
        return self.net(t_norm)
    
class ConstantParamsPinn(nn.Module):
     def __init__(self, log_Q_init, log_S_init):
         super().__init__()
        # add log Q and log S as parameters (so they are non-negative and more similar in scale)
         self.log_Q = nn.Parameter(torch.tensor(LOG_Q_INIT, dtype=torch.float32))
         self.log_S = nn.Parameter(torch.tensor(LOG_S_INIT, dtype=torch.float32))

     # Add Q and S as params
     def get_Q_S(self, t_phys):
         # t_phys shape (N, 1), return same Q and S for every point
         N = t_phys.shape[0]
         Q = torch.exp(self.log_Q).expand(N, 1)
         S = torch.exp(self.log_S).expand(N, 1)
         return Q, S


class SegmentParamsPinn(nn.Module):
    


# pass in the hyperparams to set up PINN model
model = PINN(hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN)


# separate param groups so we can use different learning rates for physics params and for the internal weights
net_params   = list(model.net.parameters()) # the weights
phys_params  = [model.log_Q, model.log_S] # log Q and log S

# use Adam optimizer
optimizer = torch.optim.Adam([
    # optimize both set of params with different learning rates
    {"params": net_params,  "lr": LR_NET},
    {"params": phys_params, "lr": LR_PARAMS},
])

# use a learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-4
)


def physics_residual(model, T_col, stats):
    C_norm_pred = model(T_col)

    dC_dt_norm = torch.autograd.grad(
        C_norm_pred, T_col,
        grad_outputs=torch.ones_like(C_norm_pred),
        create_graph=True,
    )[0]

    dt = float(stats["t_max"] - stats["t_min"])
    Q = model.Q
    S = model.S

    alpha = (Q / V) * dt
    beta = (S / (V * stats["c_std"])) * dt
    C_out_norm = (C_out - stats["c_mean"]) / stats["c_std"]

    rhs = alpha * (C_out_norm - C_norm_pred) + beta
    residual = dC_dt_norm - rhs
    return residual



