"""
PINN for IAQ CO2 Parameter Estimation
======================================

Goal: Given noisy CO2 measurements C_meas(t), recover the unknown physical
parameters Q (ventilation rate, m^3/h) and S (source term, ppm·m^3/h) from:

    V dC/dt = Q (C_out - C) + S

The PINN has TWO loss components:
  1. data_loss  — MSE between net(t) and C_meas at measurement times
  2. phys_loss  — MSE of PDE residual  [ V * dC/dt - Q*(C_out - C) - S ]²
                  evaluated on a dense grid of collocation points

Q and S are trainable scalar parameters (log-parametrised so they stay positive).

Scaling strategy
----------------
Time is normalised to [0,1].
C is normalised using mean/std computed ONLY on the training set.
Q and S are initialised from rough guesses (not the true values).
The PDE is written in normalised coordinates to avoid gradient scale mismatch.

Why log-parametrise Q and S?
------------------------------
Both must be positive and they live on very different scales
(Q~200, S~1e5). In log space a unit step is a multiplicative change,
so gradient magnitudes are comparable regardless of the raw scale.
We optimise log_Q and log_S; the physical values are exp(log_Q) etc.

Two-phase training (curriculum)
---------------------------------
Phase 1 (warm-up, ~500 steps): data_loss only.
  Lets the net fit the trend first so the PDE residual isn't computed
  against a random net at the start (which would give huge, noisy physics
  gradients that swamp the parameter signals).

Phase 2 (joint, remaining steps): data_loss + lambda_phys * phys_loss.
  Both losses active together; lambda_phys ramps up slowly so physics
  doesn't dominate before the net has a reasonable shape.

Diagnostics plotted
--------------------
  Row 1: total / data / phys loss curves over all epochs
  Row 2: recovered Q and S vs epoch  (true values shown as dashed lines)
  Row 3: predicted C(t) vs measurements  (at end of training)
  Row 4: PDE residual R(t) over the collocation grid  (should → 0)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── known / fixed physical constants ─────────────────────────────────────────
V     = 100.0   # zone volume  [m^3]  — assumed known
C_out = 420.0   # outdoor CO2  [ppm]  — assumed known
C0    = 500.0   # initial CO2  [ppm]  — assumed known (first data point)

# ── hyper-parameters ─────────────────────────────────────────────────────────
N_HIDDEN       = 3        # hidden layers
HIDDEN_DIM     = 64       # neurons per layer
N_COLLOC       = 1000     # collocation points for physics loss
LR_NET         = 3e-3     # learning rate for network weights
LR_PARAMS      = 1e-2     # (slightly higher) for log_Q, log_S
EPOCHS         = 8000     # total training epochs
WARMUP_EPOCHS  = 500      # phase-1: data loss only
LAMBDA_PHYS    = 1.0      # final physics loss weight  (ramps from 0)
RAMP_EPOCHS    = 2000     # ramp lambda_phys over this many epochs after warmup

# initial guesses — deliberately off from true values to test recovery
# true: Q≈200, S≈1e5;  we start at Q=100, S=5e4
LOG_Q_INIT = np.log(100.0)
LOG_S_INIT = np.log(5e4)

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("iaq_co2_simple.csv")

t_np = df["t_hours"].values.reshape(-1, 1).astype(np.float32)
c_np = df["C_meas_ppm"].values.reshape(-1, 1).astype(np.float32)

t_train_np, t_test_np, c_train_np, c_test_np = train_test_split(
    t_np, c_np, test_size=0.2, random_state=42
)

# ── normalisation (computed from TRAINING SET only) ───────────────────────────
t_min  = t_train_np.min()
t_max  = t_train_np.max()

c_mean = c_train_np.mean()
c_std  = c_train_np.std()

def norm_t(t):
    return (t - t_min) / (t_max - t_min + 1e-8)

def norm_c(c):
    return (c - c_mean) / c_std

def denorm_c(c_hat):
    return c_hat * c_std + c_mean

# normalised training tensors
T_train = torch.tensor(norm_t(t_train_np), dtype=torch.float32)
C_train = torch.tensor(norm_c(c_train_np), dtype=torch.float32)

# collocation points spread uniformly over the training time window
t_col_np  = np.linspace(t_min, t_max, N_COLLOC).reshape(-1, 1).astype(np.float32)
T_col     = torch.tensor(norm_t(t_col_np), dtype=torch.float32, requires_grad=True)

# ── PINN network ──────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, hidden_dim=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

        # trainable log-parameters (positive-constrained via exp)
        self.log_Q = nn.Parameter(torch.tensor(LOG_Q_INIT, dtype=torch.float32))
        self.log_S = nn.Parameter(torch.tensor(LOG_S_INIT, dtype=torch.float32))

    def forward(self, t_norm):
        """Returns normalised C prediction."""
        return self.net(t_norm)

    @property
    def Q(self):
        return torch.exp(self.log_Q)

    @property
    def S(self):
        return torch.exp(self.log_S)


model = PINN(hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN)

# separate param groups so we can use different LRs
net_params   = list(model.net.parameters())
phys_params  = [model.log_Q, model.log_S]

optimiser = torch.optim.Adam([
    {"params": net_params,  "lr": LR_NET},
    {"params": phys_params, "lr": LR_PARAMS},
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=EPOCHS, eta_min=1e-5
)

# ── physics residual ──────────────────────────────────────────────────────────
def physics_residual(T_col):
    """
    PDE in normalised space.

    C_norm = (C - c_mean) / c_std  →  C = C_norm * c_std + c_mean
    t_norm = (t - t_min) / (t_max - t_min)  →  t = t_norm * (t_max - t_min) + t_min

    dC/dt = (dC_norm/dt_norm) * (c_std / (t_max - t_min))

    PDE: V dC/dt = Q*(C_out - C) + S
    In normalised form, multiply both sides by (t_max-t_min)/(V*c_std):

        dC_norm/dt_norm = (Q / V) * (t_max-t_min) * ((C_out - c_mean)/c_std - C_norm)
                        + (S / (V * c_std)) * (t_max - t_min)

    residual R = LHS - RHS  (should be 0 if PDE is satisfied)
    """
    C_norm_pred = model(T_col)                # (N,1) normalised prediction

    # automatic differentiation for dC_norm/dt_norm
    dC_dt_norm = torch.autograd.grad(
        C_norm_pred, T_col,
        grad_outputs=torch.ones_like(C_norm_pred),
        create_graph=True,
    )[0]

    dt   = float(t_max - t_min)        # seconds / time-norm
    Q    = model.Q
    S    = model.S

    # dimensionless coefficients
    alpha = (Q / V) * dt                                     # ventilation term coeff
    beta  = (S / (V * c_std)) * dt                          # source term coeff
    C_out_norm = (C_out - c_mean) / c_std                   # normalised outdoor CO2

    rhs = alpha * (C_out_norm - C_norm_pred) + beta
    residual = dC_dt_norm - rhs
    return residual


# ── training loop ─────────────────────────────────────────────────────────────
history = {
    "loss_total": [], "loss_data": [], "loss_phys": [],
    "Q": [], "S": [],
}

print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  {'Q':>8}  {'S':>10}")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):

    optimiser.zero_grad()

    # --- data loss ---
    C_pred_train = model(T_train)
    loss_data = torch.mean((C_pred_train - C_train) ** 2)

    # --- physics loss (zero during warm-up) ---
    if epoch <= WARMUP_EPOCHS:
        loss_phys   = torch.tensor(0.0)
        lam         = 0.0
    else:
        residual    = physics_residual(T_col)
        loss_phys   = torch.mean(residual ** 2)
        # raw residual shape: (N_COLLOC, 1)

        # linear ramp of lambda_phys over RAMP_EPOCHS after warm-up
        ramp_frac   = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
        lam         = LAMBDA_PHYS * ramp_frac

    loss = loss_data + lam * loss_phys
    loss.backward()
    optimiser.step()
    scheduler.step()

    # --- record ---
    with torch.no_grad():
        q_val = model.Q.item()
        s_val = model.S.item()

    history["loss_total"].append(loss.item())
    history["loss_data"].append(loss_data.item())
    history["loss_phys"].append(loss_phys.item())
    history["Q"].append(q_val)
    history["S"].append(s_val)

    if epoch % 500 == 0 or epoch == 1:
        print(f"{epoch:>6}  {loss.item():>10.4e}  {loss_data.item():>10.4e}  "
              f"{loss_phys.item():>10.4e}  {q_val:>8.2f}  {s_val:>10.1f}")

print("\nDone.")
print(f"  Recovered  Q = {model.Q.item():.2f}  m^3/h")
print(f"  Recovered  S = {model.S.item():.2e}  ppm·m^3/h")
print(f"  (S_vol implied = {model.S.item()/1e6:.4f}  m^3 CO2/h)")

# ── diagnostics plot ──────────────────────────────────────────────────────────
epochs_arr = np.arange(1, EPOCHS + 1)

# true parameter values (for reference lines only — not used in training)
Q_TRUE = 200.0
S_TRUE = 1e6 * 0.10

fig = plt.figure(figsize=(14, 14))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Row 0: loss curves ------------------------------------------------------
ax_loss = fig.add_subplot(gs[0, :])
ax_loss.semilogy(epochs_arr, history["loss_total"], lw=1.5, label="Total loss",  color="#2c3e50")
ax_loss.semilogy(epochs_arr, history["loss_data"],  lw=1.5, label="Data loss",   color="#e74c3c", ls="--")
ax_loss.semilogy(epochs_arr, history["loss_phys"],  lw=1.5, label="Phys loss",   color="#3498db", ls="--")
ax_loss.axvline(WARMUP_EPOCHS, color="gray", ls=":", lw=1, label=f"Warm-up end ({WARMUP_EPOCHS})")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss (log scale)")
ax_loss.set_title("Training Losses")
ax_loss.legend(fontsize=9)
ax_loss.grid(alpha=0.3)

# --- Row 1: Q and S convergence ----------------------------------------------
ax_Q = fig.add_subplot(gs[1, 0])
ax_Q.plot(epochs_arr, history["Q"], lw=1.5, color="#e67e22")
ax_Q.axhline(Q_TRUE, color="black", ls="--", lw=1, label=f"True Q = {Q_TRUE:.0f}")
ax_Q.set_xlabel("Epoch")
ax_Q.set_ylabel("Q  [m³/h]")
ax_Q.set_title("Recovered Q over Training")
ax_Q.legend(fontsize=9)
ax_Q.grid(alpha=0.3)

ax_S = fig.add_subplot(gs[1, 1])
ax_S.plot(epochs_arr, history["S"], lw=1.5, color="#8e44ad")
ax_S.axhline(S_TRUE, color="black", ls="--", lw=1, label=f"True S = {S_TRUE:.2e}")
ax_S.set_xlabel("Epoch")
ax_S.set_ylabel("S  [ppm·m³/h]")
ax_S.set_title("Recovered S over Training")
ax_S.legend(fontsize=9)
ax_S.grid(alpha=0.3)

# --- Row 2: predicted vs measured C(t) ---------------------------------------
ax_fit = fig.add_subplot(gs[2, :])

# full time grid for smooth prediction
t_full_np  = np.linspace(t_np.min(), t_np.max(), 500).reshape(-1, 1).astype(np.float32)
T_full     = torch.tensor(norm_t(t_full_np), dtype=torch.float32)
with torch.no_grad():
    C_pred_norm = model(T_full).numpy()
C_pred_phys = C_pred_norm * c_std + c_mean      # denormalise

ax_fit.scatter(t_np, c_np, s=6, alpha=0.35, color="#888", label="C_meas (all data)")
ax_fit.scatter(t_train_np, c_train_np, s=8, alpha=0.5, color="#e74c3c", label="C_meas (train)")
ax_fit.plot(t_full_np, C_pred_phys, lw=2, color="#2c3e50", label="PINN prediction")

# analytical solution with true params as visual reference
C_ss_true = C_out + S_TRUE / Q_TRUE
tau_true   = V / Q_TRUE
t_ref      = t_full_np.flatten()
C_analytic = C_ss_true + (C0 - C_ss_true) * np.exp(-Q_TRUE * t_ref / V)
ax_fit.plot(t_ref, C_analytic, "--", color="#27ae60", lw=1.5, label="Analytical (true params)")

ax_fit.set_xlabel("Time [h]")
ax_fit.set_ylabel("CO₂ [ppm]")
ax_fit.set_title("PINN Prediction vs Measurements")
ax_fit.legend(fontsize=9)
ax_fit.grid(alpha=0.3)

# --- Row 3: PDE residual over collocation grid --------------------------------
ax_res = fig.add_subplot(gs[3, :])

T_col_eval = torch.tensor(norm_t(t_col_np), dtype=torch.float32, requires_grad=True)
residual_eval = physics_residual(T_col_eval)
res_np = residual_eval.detach().numpy().flatten()

ax_res.plot(t_col_np.flatten(), res_np, lw=1, color="#c0392b", alpha=0.8)
ax_res.axhline(0, color="black", lw=0.8, ls="--")
ax_res.fill_between(t_col_np.flatten(), res_np, alpha=0.15, color="#c0392b")
ax_res.set_xlabel("Time [h]")
ax_res.set_ylabel("PDE Residual (normalised)")
ax_res.set_title("Physics Residual  V dC/dt − Q(C_out−C) − S  [should → 0]")
ax_res.grid(alpha=0.3)

# --- final param box ----------------------------------------------------------
textstr = (
    f"Recovered:  Q = {model.Q.item():.1f}  (true {Q_TRUE:.0f})  m³/h\n"
    f"            S = {model.S.item():.3e}  (true {S_TRUE:.2e})  ppm·m³/h\n"
    f"        S_vol = {model.S.item()/1e6:.4f}  (true 0.1000)  m³/h"
)
fig.text(0.5, 0.01, textstr, ha="center", va="bottom", fontsize=10,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("PINN Diagnostics — IAQ CO₂ Parameter Recovery", fontsize=14, y=1.01)

OUT = Path(".")
fig.savefig(OUT / "iaq_pinn_diagnostics.png", dpi=130, bbox_inches="tight")
print(f"\nSaved figure → iaq_pinn_diagnostics.png")
plt.show()