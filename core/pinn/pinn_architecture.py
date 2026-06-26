"""Pinn_architecture.py is a module containing components and functions of pinn required for training
different variants of pinn used for a variety of situations. 
All other pinn variants are made from components from this file."""

# ------- imports --------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod

#----- PINN class -----
class PINN(nn.Module):
    """PINN consists of a neural network and a parameter model that holds Q, S parameters 
    to be optimized for. The parameter model is different depending on variant of situation."""
    def __init__(self, net, param_model):
        super().__init__()
        self.net = net # network
        self.param_model = param_model # parameter model

    @property
    def params(self):
        return self.param_model
    
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
            # nn.Linear(in_feature_number, out_feature_number)

        # make output layer and add it in to layers list
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers) # sequentially stacking the layers, like the usual set up
        # use * to unpack the list into arguments

    # get normalized C prediction
    def forward(self, t_norm):
        return self.net(t_norm)
    
class ParamModel(nn.Module, ABC):

    @abstractmethod
    def get_Q_S(self, t_phys: Tensor) -> tuple[Tensor, Tensor]:
        """
        Called inside physics_residual during training.
    Takes t_phys tensor shape (N, 1), returns Q and S tensors shape (N, 1).
        """
        pass

    @abstractmethod
    def get_final_estimates(self) -> dict:
        """
        Called after training to extract learned parameter values.
        Always returns same keys:
            Q:    np.ndarray — shape (1,) for constant, (n_segments,) for segment, (K+1,) for multi_sigmoid
            S:    np.ndarray — same shapes as Q
            taus: np.ndarray shape (K,) for multi_sigmoid, None otherwise
        """
        pass


class ConstantParams(ParamModel):
     """Constant parameter model that assumes Q and S are constants, and only adding those to the network.
     This parameter model control Q and S, which are to be optimized for."""
     def __init__(self, log_Q_init, log_S_init):
         super().__init__()
        # add log Q and log S as parameters (so they are non-negative and more similar in scale)
         self.log_Q = nn.Parameter(torch.tensor(log_Q_init, dtype=torch.float32))
         self.log_S = nn.Parameter(torch.tensor(log_S_init, dtype=torch.float32))

     # Given un-normalized time array (t_phys), return Q and S for every time point
     def get_Q_S(self, t_phys: Tensor) -> tuple[Tensor, Tensor]:
         # t_phys shape (N, 1), return same Q and S for every point
         N = t_phys.shape[0]
         Q = torch.exp(self.log_Q).expand(N, 1)
         S = torch.exp(self.log_S).expand(N, 1)
         return Q, S
        # return tensors that represent vectors, (N, 1), Q and S
    
     def get_final_estimates(self) -> dict:
         return {
             #.item changes tensor to float
             "Q": np.array([torch.exp(self.log_Q).item()]), # shape (1,)
             "S": np.array([torch.exp(self.log_S).item()]),
             "taus": None # indicate this does not exist for this variant
        }


class SegmentParams(ParamModel):
    """Segment parameter model that assumes Q or S are constant piecewise functions, and adding each segment constant
    to the network. This parameter model control all the Q or S segments, when the other is constant, 
    it simply takes same number as the value for all segments. All segments start initial guess from log_Q_init and log_S_init"""
    def __init__(self, n_segments, log_Q_init, log_S_init, segment_duration):
        super().__init__()
        self.n_segments = n_segments
        self.segment_duration = segment_duration
        self.log_Q = nn.Parameter(
            # creates a tensor of shape (n_segments, ) so like an array, and fill it with log_Q_init value for each number
            torch.full((n_segments,), log_Q_init, dtype=torch.float32)
        )
        self.log_S   = nn.Parameter(
            # creates a tensor of shape (n_segments, ) so like an array, and fill it with log_S_init value for each number
            torch.full((n_segments,), log_S_init, dtype=torch.float32)
        )

    def get_Q_S(self, t_phys):
        # calculate index from t_phys array and segment duration transferred to long, cap it at min 0 max self.n_segments - 1
        # to prevent indexing error in list
        seg_idx = torch.clamp((t_phys / self.segment_duration).long(), 0, self.n_segments - 1).squeeze(1) # removing a dimension
        Q = torch.exp(self.log_Q)[seg_idx].unsqueeze(1)
        S = torch.exp(self.log_S)[seg_idx].unsqueeze(1)
        return Q, S
    # return tensors that represent vectors, (N, 1), Q and S

    def get_final_estimates(self) -> dict:
        return {
            # .detach() disconnectes tensor from computation graph and then
            # .numpy() converts it to numpy
            "Q": torch.exp(self.log_Q).detach().numpy(), # shape (n_segments,)
            "S": torch.exp(self.log_S).detach().numpy(), # same shape as Q
            "taus": None
        }



class MultiSigmoidChangepoint(ParamModel):
    def __init__(self, t_min, t_max, tau_inits, log_Q_init, log_S_init, kappa=50.0):
        """
        Used for one single RAA-PINN that covers entire time domain in stage 2.
        tau_inits : list or array of K initial changepoint times (from Stage I peaks)
        """
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.kappa = kappa # how sharp the transition is between segments
        K = len(tau_inits) # number of changepoints
        self.tau_inits = tau_inits

        # initialise etas so sigmoid maps to each Stage I peak time
        eta_inits = [ np.log((tau - t_min) / (t_max - tau + 1e-8)) for tau in tau_inits]
        # log to invert sigmoid, tau-tim/tmax-tau+1e-8 for normalize

        self.etas = nn.ParameterList([
            nn.Parameter(torch.tensor(e, dtype=torch.float32))
            for e in eta_inits
        ]) # make each eta a tensor and add into list of parameter to optimize

        # K+1 segment values
        self.log_Q = nn.ParameterList([
            nn.Parameter(torch.tensor(log_Q_init, dtype=torch.float32)) # initialize tensors with log_Q_init as initial value
            for _ in range(K + 1) # add all the parameters of segment Q in to the parameter list
        ])
        self.log_S = nn.ParameterList([
            nn.Parameter(torch.tensor(log_S_init, dtype=torch.float32)) # initialize tensors with log_S_init as initial value
            for _ in range(K + 1) # add all the parameters of segment S in to the parameter list
        ])

    @property
    def taus(self):
        return torch.stack([
            self.t_min + (self.t_max - self.t_min) * torch.sigmoid(eta) # create taus from etas
            for eta in self.etas # do this for each eta and return the tensor that contains all the tau tensors
        ])

    def get_Q_S(self, t_phys):
        taus = self.taus  # get changepoints, shape (K,)
        Q_vals = torch.stack([torch.exp(lq) for lq in self.log_Q]) # (K+1,) ; converting to actual Q values from log
        S_vals = torch.stack([torch.exp(ls) for ls in self.log_S]) # (K+1,)

        # start at first segment, add jumps at each tau
        gates = torch.sigmoid(
            self.kappa * (t_phys - taus.unsqueeze(0))
        )  # (N, K)

        # from [a, b, c, d], get [b, c, d] and [a, b, c], then subtrct them to get differences
        dQ = Q_vals[1:] - Q_vals[:-1] # (K,) # calculate differences between each Q step
        dS = S_vals[1:] - S_vals[:-1] # (K,) # calculate differences between each S step

        Q = Q_vals[0] + (gates*dQ).sum(dim=1, keepdim=True) # make the Q tensor by adding all the differences each time
        S = S_vals[0] + (gates*dS).sum(dim=1, keepdim=True)
        return Q, S
        # return tensors that represent vectors, (N, 1), Q and S
    
    def get_final_estimates(self) -> dict:
        return {
            "Q": torch.stack(list(self.log_Q)).exp().detach().numpy(), # stack then exponent
            "S": torch.stack(list(self.log_S)).exp().detach().numpy(), 
            "taus": self.taus.detach().numpy() 
        }
    

def train_loop(model, opt_net, opt_params, sched_net, sched_params,
               T_train, C_train, T_col, stats,
               epochs, warmup_epochs, lambda_phys, ramp_epochs,
               history,
               physics_kwargs=None,
               log_fn=None):

    """
    Generic training loop for PINN, including warm-up stage, ramp, auto_scale, loss combination, backward, step.

    ### Model Training Specific Arguments
    model: the feedforward network, used to get predictions
    opt_net: the adam optimizer for network weights
    opt_params: the adam optimizer for physical parameters only
    sched_net: the cosine annealing scheduler attached to opt_net
    sched_params: the cosine annealing scheduler attached to opt_oarans
    
    ### Training data and information Arguments
    T_train: normalized time tensor
    C_train: normalized CO2 tensor
    T_col: normalized time tensor for collocation points
    stats: dictionary containing t_min, t_max, y_mean, y_std

    ### Hyperparameter Arguments
    epochs: total number of training iterations
    warmup_epochs: how many epochs to use data loss only before using incorporating physics loss
    lambda_phys: the target weight for physics loss when fully ramped up, 1.0 means physics loss is balanced with data loss
    ramp_epochs: how many epochs after warm-up to linearly increase the physics weight from 0 to its full value
    
    ### Physics Arguments
    physics_kwargs: dict of V and C_out passed into physics_residual

    ### Logging Arguments
    history: dictionary of lists that accumulate one value per epoch for loss_total, loss_data, loss_phys, and whatever log_fn appends
    log_fn: Logger function specific for the type of pinn you are making
    """
    if physics_kwargs is None:
        physics_kwargs = {}

    phys_loss_init = None
    auto_scale_frozen = None

    for epoch in range(1, epochs + 1):
        opt_net.zero_grad()  # clear old gradients
        opt_params.zero_grad()  # clear old parameters' gradients

        C_pred_train = model(T_train) # forward pass
        loss_data = torch.mean((C_pred_train - C_train)**2) # compute data loss
        
        # --- physics loss (zero during warm-up) ---
        if epoch <= warmup_epochs: # for the first warmup_epochs we don't use physics loss and only consider data loss
            loss_phys = torch.tensor(0.0, device=C_train.device) # physics loss = 0
            lam = 0.0 # lam is total weight of physics loss
        else: # afterward, we slowly ramp up how much physics loss is weighted until warmup+rampepochs number to full weight
            residual = physics_residual(model, T_col, stats, **physics_kwargs) # calculate physics residual
            loss_phys = torch.mean(residual ** 2) # calculate physics loss

            if phys_loss_init is None: # if it's the first phys epoch
                phys_loss_init = (loss_phys.detach().item())  # extract first physics epoch loss from loss phys tensor as our init phys loss
                auto_scale_frozen = (loss_data.detach().item() / (phys_loss_init + 1e-8))
            
            # if it went over 1, choose 1 as ramp frac because that's the largest weight / full weight for physics loss
            ramp_frac = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
            lam = (lambda_phys * ramp_frac * auto_scale_frozen)
         
        loss = loss_data + lam * loss_phys # calculate total loss
        loss.backward() # computes all gradients of loss
        opt_net.step() # change parameter value
        opt_params.step()
        sched_net.step() # decay learning rate according to cosine schedule
        sched_params.step()
        
        # add data to corresponding history list
        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())

        # call logging function
        if log_fn is not None:
            log_fn(model, history, epoch)

    return history


def physics_residual(model, T_col, stats, V, C_out):
    """The physics residual function"""
    C_norm_pred = model(T_col)
    dC_dt_norm = torch.autograd.grad(C_norm_pred,T_col,grad_outputs=torch.ones_like(C_norm_pred),
                                    create_graph=True,)[0]

    dt = float(stats["t_max"] - stats["t_min"])
    t_phys = (stats["t_min"] + T_col*dt)

    Q, S = model.params.get_Q_S(t_phys)

    alpha = (Q / V) * dt
    beta = (S / (V*stats["y_std"])) * dt
    C_out_norm = (C_out - stats["y_mean"]) / stats["y_std"]

    rhs = (alpha * (C_out_norm - C_norm_pred) + beta)
    residual = (dC_dt_norm - rhs)

    return residual



# class SigmoidChangepoint(ParamModel):
#     """Sigmoid changepoint parameter model assumes Q and S are no longer step functions, instead, they are sigmoid functions
#     that are differentiable. This is typically paired with RAA-PINN. This model adds 4 trainable scalars log parametrized,
#     which is used for each run of RAA-PINN stage 2 in individual segments.

#     Model guesses a changepoint tau between [t_left, t_right], and the left side value and right side value of Q, S
    
#     It looks at one small interval and try to optimize for these 4 parameters:
#     Q_minus - the ventilation rate during [t_left, tau] (estimate of Q on the left side of changepoint)
#     Q_plus - the ventilation rate during [tau, t_right] (estimate of Q on the right side of changepoint)
#     similarly for S."""
#     def __init__(self, t_left, t_right, log_Q_init, log_S_init, kappa=50.0):
#         super().__init__()
#         self.t_left = t_left # left boundary of time interval
#         self.t_right = t_right # right boundary of time interval
#         self.kappa = kappa # how sharp the transition is between before and after value

#         self.log_Q_minus = nn.Parameter(torch.tensor(log_Q_init, dtype=torch.float32))
#         self.log_Q_plus  = nn.Parameter(torch.tensor(log_Q_init, dtype=torch.float32))
#         self.log_S_minus = nn.Parameter(torch.tensor(log_S_init, dtype=torch.float32))
#         self.log_S_plus  = nn.Parameter(torch.tensor(log_S_init, dtype=torch.float32))
#         # eta = 0, starts at midpoint of the time interval, it's the percentage of where the changepoint is in the interval
#         self.eta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

#     @property
#     def tau(self): # method to get tau calculated and returned
#         # constrain changepoint to stay within interval [t_left, t_right]
#         return self.t_left + (self.t_right - self.t_left) * torch.sigmoid(self.eta)

#     def get_Q_S(self, t_phys):
#         gate  = torch.sigmoid(self.kappa * (t_phys - self.tau)) 
#         # create sigmoid function that slides Q to left/right of tau
#         Q = torch.exp(self.log_Q_minus) * (1 - gate) + torch.exp(self.log_Q_plus) * gate
#         S = torch.exp(self.log_S_minus) * (1 - gate) + torch.exp(self.log_S_plus) * gate
#         return Q, S
#         # return tensors that represent vectors, (N, 1), Q and S