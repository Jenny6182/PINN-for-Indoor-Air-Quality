"""
PINN to Estimate IAQ CO2 Parameters
-----------------------------------
PINN has 2 loss components, one is PDE/physics loss, the other is data loss
Together they make the total loss

Problem's ODE: 
V dC/dt = Q (C_out - C) + S
-> 
0 = V dC/dt - Q (C_out - C) + S   (move everything to right hand side)

when 0, means there's no residual so perfect fit, use the constant as residual

1. Physics loss:
Physics residual:
f = Q (C_out - C) + S - V dC/dt

The Q (ventilation rate, m^3/h) and S (source term, ppm·m^3/h) are lambda1, lambda2 like from the PINNs paper
Add them into trainable parameters to estimate in the nn, along with weights and biases
(log-parametrised so they are positive and can balance their scale because their scale could be very different)
(like in the case of the constants, Q is 200, S is 100000 which are very different in scale)

physics loss is the MSE of PDE loss function is f^2 = [V * dC/dt - Q*(C_out - C) - S ]^2
evaluated on dense collocation points made from the ODE

2. Data loss
is the MSE between net(t) predicted by nn and C_meas at measurement times

--------
Few problems and design choices
- scaling (nn perform better with normalized input and output so time is normalized from [0, 1], C uses std to normalize)
- Q and S are on difference scales, and they must both be positive (?)
    -> so log parametrized, and optimize log Q and log S then recover with exponential later
- need to warm up first without physics 
    -> used only data to fit for short amount of time so PDE residual isn't super large and overwhelm gradient
    -> first 500 epochs don't move Q and S while optimizing just based on data loss
- slowly ramp up how much physics loss is weighted until lambda_phys/ramp_epochs fraction become 1
    -> same reason as above, so when physics term come into play it doesn't suddenly overwhelm gradient
       and cause random behavior

"""

# ------- imports --------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from plotting import plot_all_diagnostics

# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
SEED = 42
torch.manual_seed(SEED) # set random numbers in torch
np.random.seed(SEED) # in numpy

# known physical constants 
V = 100.0   # zone volume, m^3
C_out = 420.0   # outdoor CO2, ppm
S_vol = 0.10
C0 = 500.0   # initial CO2, ppm  (first data point)
Q = 400 # is not used, just for reference
S = 1e6 * S_vol # is not used, just for reference

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

# tested value pairs: Q=100, S=5e4; 10, 3e4; 10, 1e4; 1, 1e4; 1, 1
# all were able to achieve pretty good result that's close to true values

# ------ load data ------
df = pd.read_csv("varying_datasets/iaq_co2_varying_Q.csv") # read from csv

# get the columns input and output into numpy arrays from pandas dfs
t_np = df["t_hours"].values.reshape(-1, 1).astype(np.float32)
c_np = df["C_meas_ppm"].values.reshape(-1, 1).astype(np.float32)

# split training and testing cuz small dataset so no validation set
t_train_np, t_test_np, c_train_np, c_test_np = train_test_split(
    t_np, c_np, test_size=0.2, random_state=42
)

# ------ normalisation (getting min max std mean of time and co2 only from training data to avoid data leak)
# usually nn exxpect input from 0 to 1 to be stable (cuz gradient could be very big) so we'll normalize time too
t_min  = t_train_np.min() 
t_max  = t_train_np.max()

c_mean = c_train_np.mean()
c_std  = c_train_np.std()

# Normalize formula:
# x-x_min / x_max-x_min or x-x_min / x_std bc trying to make x_std = 1

# Time is normalised to [0,1].
def norm_t(t,  t_min, t_max):
    return (t - t_min) / (t_max - t_min + 1e-8)
    # adding 1e-8 a very small number to prevent division by zero in the future

def norm_c(c):
    return (c - c_mean) / c_std

# To reverse normalization, multiply by std and add mean
def denorm_c(c_hat):
    return c_hat * c_std + c_mean

# normalize training tensors
T_train = torch.tensor(norm_t(t_train_np, t_min, t_max), dtype=torch.float32)
C_train = torch.tensor(norm_c(c_train_np), dtype=torch.float32)

# ------ make collocation points that should spread uniformly over the training time window ------
t_col_np = np.linspace(t_min, t_max, N_COLLOC).reshape(-1, 1).astype(np.float32) # make np list
T_col = torch.tensor(norm_t(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True) # turn np list into tensor, requires gradient=True to compute dC/dt


# ----- set up PINN ------
# make a class that inherit from nn.Module
class PINN(nn.Module):
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

        # add log Q and log S as parameters (so they are non-negative and more similar in scale)
        self.log_Q = nn.Parameter(torch.tensor(LOG_Q_INIT, dtype=torch.float32))
        self.log_S = nn.Parameter(torch.tensor(LOG_S_INIT, dtype=torch.float32))

    # get normalized C prediction
    def forward(self, t_norm):
        return self.net(t_norm)

    # Add Q and S as params
    @property
    def Q(self):
        return torch.exp(self.log_Q)

    @property
    def S(self):
        return torch.exp(self.log_S)

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

# ---- physics residual ----

def physics_residual(model, T_col, t_max, t_min, c_std, c_mean):
    C_norm_pred = model(T_col)

    dC_dt_norm = torch.autograd.grad(
        C_norm_pred, T_col,
        grad_outputs=torch.ones_like(C_norm_pred),
        create_graph=True,
    )[0]

    dt = float(t_max - t_min)
    Q = model.Q
    S = model.S

    alpha = (Q / V) * dt
    beta = (S / (V * c_std)) * dt
    C_out_norm = (C_out - c_mean) / c_std

    rhs = alpha * (C_out_norm - C_norm_pred) + beta
    residual = dC_dt_norm - rhs
    return residual

# def physics_residual(T_col):
#     # doing everything normalized

#     C_norm_pred = model(T_col) # (N,1) normalised prediction

#     # automatic differentiation for dC_norm/dt_norm
#     dC_dt_norm = torch.autograd.grad(
#         C_norm_pred, T_col,
#         grad_outputs=torch.ones_like(C_norm_pred),
#         create_graph=True,
#     )[0]

#     dt = float(t_max - t_min) #  time norm
#     Q = model.Q
#     S = model.S

#     # dimensionless coefficients
#     alpha = (Q / V) * dt   # ventilation term coeff
#     beta = (S / (V * c_std)) * dt    # source term coeff
#     C_out_norm = (C_out - c_mean) / c_std   # normalised outdoor CO2

#     # right hand side equation
#     rhs = alpha * (C_out_norm - C_norm_pred) + beta
#     residual = dC_dt_norm - rhs
#     return residual


# ----- training -----  
# put data into dictionary, this dictionary is used for plotting later on
history = {
    "loss_total": [], "loss_data": [], "loss_phys": [],
    "Q": [], "S": [],
}

# printing table title, will report these numbers at every 500 epoch
print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  {'Q':>8}  {'S':>10}")
print("-" * 60)

# Using this to calculate ramp up fraction for the physics loss weight
phys_loss_init = None

# Training loop
for epoch in range(1, EPOCHS + 1):

    optimizer.zero_grad() # clear old gradients

    # --- data loss ---
    C_pred_train = model(T_train)
    loss_data = torch.mean((C_pred_train - C_train) ** 2)

    # --- physics loss (zero during warm-up) ---
    if epoch <= WARMUP_EPOCHS: # for the first warmup_epochs we don't use physics loss and only consider data loss
        loss_phys = torch.tensor(0.0) # physics loss = 0
        lam = 0.0
    else: # afterward, we slowly ramp up how much physics loss is weighted until warmup+rampepochs number to full weight
        residual  = physics_residual(T_col)
        loss_phys = torch.mean(residual ** 2)
        auto_scale_full = -1

        if epoch == WARMUP_EPOCHS + 1: # if it's the first phys epoch
            phys_loss_init = loss_phys.detach().item() # extract first physics epoch loss from loss phys tensor as our init phys loss
            auto_scale_full = loss_data.detach().item() / (phys_loss_init + 1e-8)

        # if it went over 1, choose 1 as ramp frac because that's the largest weight / full weight for physics loss
        ramp_frac  = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)

        # this is for normalizing scale difference between two loss terms so not one of them dominate
        # so they both contribute to gradient fairly
        auto_scale = loss_data.detach().item() / (phys_loss_init + 1e-8) # current data loss / initial physics loss 

        if auto_scale_full != -1:
            lam = LAMBDA_PHYS * ramp_frac * auto_scale_full # full weight
        else:
            lam = LAMBDA_PHYS * ramp_frac * auto_scale  # the actual weighting coefficient applied to physics loss
        
        # it's the weight we should give physics loss right now * the normalizing scale factor
        # I set lambda phys to 1 so no effect

    # total loss (objective to minimize) is data loss + the weighted physics loss
    loss = loss_data + lam * loss_phys

    loss.backward() # computes all gradients of loss
    optimizer.step() # change parameter value
    scheduler.step() # decay learning rate according to cosine schedule

    # --- record ---
    with torch.no_grad(): # don't track gradients inside here cuz just reading for logging not part of training
        q_val = model.Q.item()
        s_val = model.S.item()

    # add data to corresponding history list
    history["loss_total"].append(loss.item())
    history["loss_data"].append(loss_data.item())
    history["loss_phys"].append(loss_phys.item())
    history["Q"].append(q_val)
    history["S"].append(s_val)

    # every 500 epoch, print a message to show progress
    if epoch % 500 == 0 or epoch == 1:
        print(f"{epoch:>6}  {loss.item():>10.4e}  {loss_data.item():>10.4e}  "
              f"{loss_phys.item():>10.4e}  {q_val:>8.2f}  {s_val:>10.1f}")

print("\nDone.")
print(f"  Estimated Q = {model.Q.item():.2f}  m^3/h")
print(f"  Estimated S = {model.S.item():.2e}  ppm·m^3/h")
print(f"  (S_vol implied = {model.S.item()/1e6:.4f}  m^3 CO2/h)")

# --- Generate diagnostic plots ---
print(Q)
print(S)
plot_all_diagnostics(
    model=model,
    history=history,
    t_np=t_np,
    c_np=c_np,
    t_train_np=t_train_np,
    c_train_np=c_train_np,
    t_col_np=t_col_np,
    physics_residual_fn=physics_residual,
    norm_t=norm_t,
    epochs=EPOCHS,
    warmup_epochs=WARMUP_EPOCHS,
    t_min=t_min,
    t_max=t_max,
    c_mean=c_mean,
    c_std=c_std,
    C_out=C_out,
    V=V,
    C0=C0,
    Q_TRUE=Q,
    S_TRUE=S,
    output_path="iaq_pinn_diagnostics.png"
)