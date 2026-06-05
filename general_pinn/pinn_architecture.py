

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
class PINN(nn.Module):
    """PINN consists of a neural network and a parameter model that holds Q, S parameters to be optimized for, which is different for each situation."""
    def __init__(self, net, param_model):
        super().__init__()
        self.net = net # network
        self.param_model = param_model # parameter model

    def forward(self, t_norm): # forward pass
        return self.net(t_norm)


# ----- set up PINN ------
# make a class that inherit from nn.Module
class FeedForwardNet(nn.Module):
    """Feed forward network component, takes normalized t and return normalized C as network output.
    The neural network is responsible for controlling the shape of C(t)"""
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
    
class ConstantParams(nn.Module):
     """Constant parameter model that assumes Q and S are constants, and only adding those to the network.
     This parameter model control Q and S, which are to be optimized for."""
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


class SegmentParams(nn.Module):
    """Segment parameter model that assumes Q or S are constant piecewise functions, and adding each segment constant
    to the network. This parameter model control all the Q or S segments, when the other is constant, 
    it simply takes same number as the value for all segments."""
    def __init__(self, n_segments, log_Q_init, log_S_init, segment_duration):
        super().__init__()
        self.n_segments       = n_segments
        self.segment_duration = segment_duration
        self.log_Q_segments   = nn.Parameter(
            torch.full((n_segments,), log_Q_init, dtype=torch.float32)
        )
        self.log_S_segments   = nn.Parameter(
            torch.full((n_segments,), log_S_init, dtype=torch.float32)
        )

    def get_Q_S(self, t_phys):
        seg_idx = torch.clamp(
            (t_phys / self.segment_duration).long(), 0, self.n_segments - 1
        ).squeeze(1)
        Q = torch.exp(self.log_Q_segments)[seg_idx].unsqueeze(1)
        S = torch.exp(self.log_S_segments)[seg_idx].unsqueeze(1)
        return Q, S

class SigmoidChangepoint(nn.Module):
    """Sigmoid changepoint parameter model assumes Q and S are no longer step functions, instead, they are sigmoid functions
    that are differentiable. This is typically paired with RAA-PINN. This model adds 4 trainable scalars log parametrized,
    which is used for each run of RAA-PINN stage 2. 
    
    It looks at one small interval and try to optimize for these 4 parameters:
    Q_minus - the ventilation rate during [t_left, tau] (estimate of Q on the left side of changepoint)
    Q_plus - the ventilation rate during [tau, t_right] (estimate of Q on the right side of changepoint)
    similarly for S."""
    def __init__(self, t_left, t_right, log_Q_init, log_S_init, kappa=50.0):
        super().__init__()
        self.t_left  = t_left
        self.t_right = t_right
        self.kappa   = kappa

        self.log_Q_minus = nn.Parameter(torch.tensor(log_Q_init, dtype=torch.float32))
        self.log_Q_plus  = nn.Parameter(torch.tensor(log_Q_init, dtype=torch.float32))
        self.log_S_minus = nn.Parameter(torch.tensor(log_S_init, dtype=torch.float32))
        self.log_S_plus  = nn.Parameter(torch.tensor(log_S_init, dtype=torch.float32))
        # eta=0 maps to midpoint of interval — natural starting point
        self.eta         = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def tau(self):
        # constrain changepoint to stay inside (t_left, t_right)
        return self.t_left + (self.t_right - self.t_left) * torch.sigmoid(self.eta)

    def get_Q_S(self, t_phys):
        gate  = torch.sigmoid(self.kappa * (t_phys - self.tau))
        Q     = torch.exp(self.log_Q_minus) * (1 - gate) + torch.exp(self.log_Q_plus) * gate
        S     = torch.exp(self.log_S_minus) * (1 - gate) + torch.exp(self.log_S_plus) * gate
        return Q, S
    

def train_loop(model, opt_net, opt_params, sched_net, sched_params,
               T_train, C_train, T_col, stats,
               epochs, warmup_epochs, lambda_phys, ramp_epochs,
               history, log_fn=None):
    
    """The training loop for PINN: includes warm-up stage, ramp, auto_scale, loss combination, backward, step.
    model: the feedforward network, used to get predictions
    opt_net: the adam optimizer for network weights
    opt_params: the adam optimizer for physical parameters only
    sched_net: the cosine annealing scheduler attached to opt_net
    sched_params: the cosine annealing scheduler attached to opt_oarans
    T_train: normalized time tensor
    C_train: normalized CO2 tensor
    T_col: normalized time tensor for collocation points
    stats: dictionary containing t_min, t_max, y_mean, y_std
    epochs: total number of training iterations
    warmup_epochs: how many epochs to use data loss only before using incorporating physics loss
    lambda_phys: the target weight for physics loss when fully ramped up, 1.0 means physics loss is balanced with data loss
    ramp_epochs: how many epochs after warm-up to linearly increase the physics weight from 0 to its full value
    history: dictionary of lists that accumulate one value per epoch for loss_total, loss_data, loss_phys, and whatever log_fn appends
    log_fn: optional callback function with signature log_fn(model, history, epoch), called every epoch to append PINN-specific values to history"""

    t_min  = stats["t_min"]
    t_max  = stats["t_max"]
    c_mean = stats["y_mean"]
    c_std  = stats["y_std"]

    phys_loss_init    = None
    auto_scale_frozen = None

    for epoch in range(1, epochs + 1):
        opt_net.zero_grad()
        opt_params.zero_grad()

        C_pred_train = model(T_train)
        loss_data    = torch.mean((C_pred_train - C_train) ** 2)

        if epoch <= warmup_epochs:
            loss_phys = torch.tensor(0.0)
            lam       = 0.0
        else:
            residual  = physics_residual(model, T_col, t_min, t_max, c_mean, c_std)
            loss_phys = torch.mean(residual ** 2)

            if phys_loss_init is None:
                phys_loss_init    = loss_phys.detach().item()
                auto_scale_frozen = loss_data.detach().item() / (phys_loss_init + 1e-8)

            ramp_frac = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
            lam       = lambda_phys * ramp_frac * auto_scale_frozen

        loss = loss_data + lam * loss_phys
        loss.backward()
        opt_net.step()
        opt_params.step()
        sched_net.step()
        sched_params.step()

        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())

        # log_fn extracts whatever extra values are relevant for this PINN
        if log_fn is not None:
            log_fn(model, history, epoch)

    return history


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



