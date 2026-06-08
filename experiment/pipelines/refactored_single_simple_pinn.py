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
from core.utils.plotting import plot_all_simple

# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# known physical constants 
V = 100.0       # zone volume, m^3
C_out = 420.0   # outdoor CO2, ppm
S_vol = 0.10
C0 = 500.0      # initial CO2, ppm (first data point)

# ------ hyper-parameters ------
N_HIDDEN = 3        # number of hidden layers
HIDDEN_DIM = 64     # number of neurons per layer
N_COLLOC = 1000     # number of collocation points
LR_NET = 3e-3       # learning rate for network weights
LR_PARAMS = 1e-2    # learning rate for log_Q, log_S parameters
EPOCHS = 8000       # total training epochs

WARMUP_EPOCHS = 500     # how many epochs to use only data loss
LAMBDA_PHYS = 1.0       # final physics loss weight
RAMP_EPOCHS = 2000      # ramp lambda_phys over this many epochs after warmup

# initial guesses of Q and S (log-parametrized)
LOG_Q_INIT = np.log(1)
LOG_S_INIT = np.log(1)


def load_data(path):
    """path ex: 'datasets/iaq_Q400_S100000.csv'"""
    df = pd.read_csv(path)

    t_np = df["t_hours"].values.reshape(-1, 1).astype(np.float32)
    c_np = df["C_meas_ppm"].values.reshape(-1, 1).astype(np.float32)

    t_train_np, t_test_np, c_train_np, c_test_np = train_test_split(
        t_np, c_np, test_size=0.2, random_state=42
    )

    return t_np, c_np, t_train_np, c_train_np, t_test_np, c_test_np


def norm_t(t, t_min, t_max):
    return (t - t_min) / (t_max - t_min + 1e-8)

def norm_c(c, c_std, c_mean):
    return (c - c_mean) / c_std

def denorm_c(c_hat, c_std, c_mean):
    return c_hat * c_std + c_mean


def normalize_data(t_train_np, c_train_np, t_min, t_max, c_std, c_mean):
    T_train_norm = torch.tensor(norm_t(t_train_np, t_min, t_max), dtype=torch.float32)
    C_train_norm = torch.tensor(norm_c(c_train_np, c_std, c_mean), dtype=torch.float32)
    return T_train_norm, C_train_norm

def create_collocation_points(t_min, t_max, N_COLLOC=N_COLLOC):
    t_col_np = np.linspace(t_min, t_max, N_COLLOC).reshape(-1, 1).astype(np.float32)
    T_col = torch.tensor(norm_t(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True)
    return t_col_np, T_col

def find_statistical_elements(t_train_np, c_train_np):
    t_min  = t_train_np.min()
    t_max  = t_train_np.max()
    c_std  = c_train_np.std()
    c_mean = c_train_np.mean()
    return t_min, t_max, c_std, c_mean


# ----- PINN model -----
class PINN(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden=3):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers)

        self.log_Q = nn.Parameter(torch.tensor(LOG_Q_INIT, dtype=torch.float32))
        self.log_S = nn.Parameter(torch.tensor(LOG_S_INIT, dtype=torch.float32))

    def forward(self, t_norm):
        return self.net(t_norm)

    @property
    def Q(self):
        return torch.exp(self.log_Q)

    @property
    def S(self):
        return torch.exp(self.log_S)


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


def train_loop(model, optimizer, scheduler, T_train, C_train, T_col,
               t_max, t_min, c_std, c_mean, history):
    """
    Runs the full training loop and populates `history` in-place.
    Returns the filled history dict.
    """
    # FIX: phys_loss_init lives here, not at module level
    phys_loss_init = None

    print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  {'Q':>8}  {'S':>10}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad()

        # --- data loss ---
        C_pred_train = model(T_train)
        loss_data = torch.mean((C_pred_train - C_train) ** 2)

        # --- physics loss ---
        if epoch <= WARMUP_EPOCHS:
            loss_phys = torch.tensor(0.0)
            lam = 0.0
        else:
            residual  = physics_residual(model, T_col, t_max, t_min, c_std, c_mean)
            loss_phys = torch.mean(residual ** 2)

            # FIX: capture initial physics loss on the very first physics epoch
            if phys_loss_init is None:
                phys_loss_init = loss_phys.detach().item()
                auto_scale_full = loss_data.detach().item() / (phys_loss_init + 1e-8)
            else:
                auto_scale_full = None

            ramp_frac  = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
            auto_scale = loss_data.detach().item() / (phys_loss_init + 1e-8)

            if auto_scale_full is not None:
                lam = LAMBDA_PHYS * ramp_frac * auto_scale_full
            else:
                lam = LAMBDA_PHYS * ramp_frac * auto_scale

        loss = loss_data + lam * loss_phys

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            q_val = model.Q.item()
            s_val = model.S.item()

        # FIX: write into the history dict that was passed in
        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())
        history["Q"].append(q_val)
        history["S"].append(s_val)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>10.4e}  {loss_data.item():>10.4e}  "
                  f"{loss_phys.item():>10.4e}  {q_val:>8.2f}  {s_val:>10.1f}")

    return history


def parse_true_params(path):
    """
    Extract true Q and S from a filename like 'iaq_Q800_S0.05.csv'.
    The S value in the filename is S_vol [m^3 CO2/h].
    The ODE uses S in [ppm * m^3/h] = S_vol * 1e6.
    Returns (Q_true, S_true_ode, S_vol_true).
    """
    stem   = Path(path).stem         # e.g. 'iaq_Q800_S0.05'
    parts  = stem.split("_")         # ['iaq', 'Q800', 'S0.05']
    Q      = float(parts[1][1:])     # 800.0  (strip leading 'Q')
    S_vol  = float(parts[2][1:])     # 0.05   (strip leading 'S') — this is S_vol [m^3 CO2/h]
    S_ode  = S_vol * 1e6             # convert to ODE units: ppm * m^3/h
    return Q, S_ode, S_vol


def main(path, plot_output_path=None, Q_true=None, S_true=None):
    # FIX: use Path.stem so the extension is never included in S
    if Q_true is not None:
        print(f"True Q={Q_true}")
    
    if S_true is not None:
        print(f"True Q={Q_true}")

    # Q_true, S_true, S_vol_true = parse_true_params(path)
    # print(f"File: {path}  |  True Q={Q_true}, True S_vol={S_vol_true}, True S={S_true:.2e}")
    print(f"File: {path} ")

    # load data
    t_np, c_np, t_train_np, c_train_np, t_test_np, c_test_np = load_data(path)

    t_min, t_max, c_std, c_mean = find_statistical_elements(t_train_np, c_train_np)

    T_train, C_train = normalize_data(t_train_np, c_train_np, t_min, t_max, c_std, c_mean)

    t_col_np, T_col = create_collocation_points(t_min, t_max, N_COLLOC=N_COLLOC)

    # build model
    model = PINN(hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN)

    net_params  = list(model.net.parameters())
    phys_params = [model.log_Q, model.log_S]

    optimizer = torch.optim.Adam([
        {"params": net_params,  "lr": LR_NET},
        {"params": phys_params, "lr": LR_PARAMS},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4
    )

    # FIX: create history here and pass it into train_loop
    history = {
        "loss_total": [], "loss_data": [], "loss_phys": [],
        "Q": [], "S": [],
    }

    history = train_loop(
        model, optimizer, scheduler,
        T_train, C_train, T_col,
        t_max, t_min, c_std, c_mean,
        history,           # <-- passed in, filled in-place, returned
    )

    Q_est = model.Q.item()
    S_est = model.S.item()

    print("\nDone.")
    print(f"  Estimated Q = {Q_est:.2f}  m^3/h")
    print(f"  Estimated S = {S_est:.2e}  ppm·m^3/h")
    print(f"  (S_vol implied = {S_est/1e6:.4f}  m^3 CO2/h)")

    # use caller-supplied path if given, otherwise default to local dir
    stem = Path(path).stem
    output_path = plot_output_path or f"iaq_pinn_diagnostics_{stem}.png"

    plot_all_simple(
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
        Q_TRUE=Q_true,
        S_TRUE=S_true,
        output_path=output_path,
    )

    # Return key numbers so the caller (batch_train) can log them
    return {
        "file":        Path(path).name,
        "Q_true":      Q_true,
        # "S_vol_true":  S_vol_true,
        "S_true":      S_true,
        "Q_est":       round(Q_est, 4),
        "S_est":       round(S_est, 4),
        "S_vol_est":   round(S_est / 1e6, 6),
        "final_loss_total": round(history["loss_total"][-1], 6),
        "final_loss_data":  round(history["loss_data"][-1],  6),
        "final_loss_phys":  round(history["loss_phys"][-1],  6),
    }