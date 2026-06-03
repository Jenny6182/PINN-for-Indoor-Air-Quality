import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys


# ============================================================
# Constants
# ============================================================

V = 100.0
C_out = 420.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Data
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    t_np = df["t_hours"].values.reshape(-1, 1).astype(np.float32)
    c_np = df["C_meas_ppm"].values.reshape(-1, 1).astype(np.float32)

    t_train, t_test, c_train, c_test = train_test_split(
        t_np, c_np, test_size=0.2, random_state=42
    )

    return t_np, c_np, t_train, c_train, t_test, c_test


# ============================================================
# Normalization
# ============================================================

def norm_t(t, t_min, t_max):
    return (t - t_min) / (t_max - t_min + 1e-8)


def norm_c(c, mean, std):
    return (c - mean) / (std + 1e-8)


# ============================================================
# Model
# ============================================================

class PINN(nn.Module):
    def __init__(self, n_segments):
        super().__init__()

        self.K = n_segments - 1

        # C(t)
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # piecewise parameters
        self.log_Q = nn.Parameter(torch.zeros(n_segments))
        self.log_S = nn.Parameter(torch.zeros(n_segments))

        # changepoints (raw → ordered via softplus + cumulative sum)
        self.raw_tau = nn.Parameter(torch.randn(self.K) * 0.1)

        self.kappa = 20.0

    def forward(self, t):
        return self.net(t)

    @property
    def Q(self):
        return torch.exp(self.log_Q)

    @property
    def S(self):
        return torch.exp(self.log_S)

    def tau(self, t_min, t_max):
        delta = torch.nn.functional.softplus(self.raw_tau)
        delta = delta / (delta.sum() + 1e-8)
        return t_min + torch.cumsum(delta, dim=0) * (t_max - t_min)


# ============================================================
# Piecewise construction
# ============================================================

def build_piecewise(model, t, t_min, t_max):

    tau = model.tau(t_min, t_max)
    Q = model.Q
    S = model.S

    kappa = model.kappa

    g = torch.sigmoid(kappa * (t - tau))  # (N, K)

    w = []
    w.append(1 - g[:, 0:1])

    for i in range(1, len(tau)):
        w.append(g[:, i-1:i] - g[:, i:i+1])

    w.append(g[:, -1:])

    w = torch.cat(w, dim=1)

    Q_t = (w * Q).sum(dim=1, keepdim=True)
    S_t = (w * S).sum(dim=1, keepdim=True)

    return Q_t, S_t, tau


# ============================================================
# Physics residual
# ============================================================

def physics_residual(model, T, t_min, t_max, c_mean, c_std):

    C = model(T)

    dC_dt = torch.autograd.grad(
        C,
        T,
        grad_outputs=torch.ones_like(C),
        create_graph=True
    )[0]

    t_phys = T * (t_max - t_min) + t_min

    Q_t, S_t, _ = build_piecewise(model, t_phys, t_min, t_max)

    C_out_n = (C_out - c_mean) / c_std

    rhs = (Q_t / V) * (C_out_n - C) + (S_t / (V * c_std))

    return dC_dt - rhs


# ============================================================
# Training
# ============================================================
def train(model, optimizer, T_train, C_train, T_col,
          t_min, t_max, c_mean, c_std, epochs=8000):

    history = {
        "loss_total": [],
        "loss_data": [],
        "loss_phys": [],
        "Q_mean": [],
        "S_mean": [],
    }

    print(f"{'Epoch':>6} | {'Total':>10} | {'Data':>10} | {'Phys':>10} | {'Q_mean':>8} | {'S_mean':>8}")
    print("-" * 70)

    for ep in range(1, epochs + 1):

        optimizer.zero_grad()

        # data loss
        C_pred = model(T_train)
        loss_data = torch.mean((C_pred - C_train) ** 2)

        # physics loss (warmup)
        if ep < 500:
            loss_phys = torch.tensor(0.0)
        else:
            res = physics_residual(model, T_col, t_min, t_max, c_mean, c_std)
            loss_phys = torch.mean(res ** 2)

        loss = loss_data + loss_phys
        loss.backward()
        optimizer.step()

        # stats
        q_mean = model.Q.mean().item()
        s_mean = model.S.mean().item()

        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())
        history["Q_mean"].append(q_mean)
        history["S_mean"].append(s_mean)

        # live print (same style as your old version)
        if ep % 500 == 0 or ep == 1:
            print(f"{ep:6d} | {loss.item():10.4e} | {loss_data.item():10.4e} | "
                  f"{loss_phys.item():10.4e} | {q_mean:8.2f} | {s_mean:8.2e}")

    return history


# ============================================================
# Plotting helpers (COMPATIBLE WITH YOUR EARLIER FILE)
# ============================================================


def plot_predictions_vs_measurements(ax, model, t_np, c_np, t_train_np, c_train_np,
                                     t_min, t_max, c_mean, c_std, norm_t,
                                     C_out, S_TRUE, Q_TRUE, V, C0):
    """Plot PINN predictions vs measured data with analytical solution."""
    # Full time grid for smooth prediction
    t_full_np = np.linspace(t_np.min(), t_np.max(), 500).reshape(-1, 1).astype(np.float32)
    T_full = torch.tensor(norm_t(t_full_np, t_min, t_max), dtype=torch.float32)

    with torch.no_grad():
        C_pred_norm = model(T_full).numpy()
    C_pred_phys = C_pred_norm * c_std + c_mean  # denormalize

    ax.scatter(t_np,       c_np,       s=6, alpha=0.35, color="#888",    label="C_meas (all data)")
    ax.scatter(t_train_np, c_train_np, s=8, alpha=0.5,  color="#e74c3c", label="C_meas (train)")
    ax.plot(t_full_np, C_pred_phys, lw=2, color="#2c3e50", label="PINN prediction")

    # Analytical solution using true params as a visual reference
    t_ref     = t_full_np.flatten()
    C_ss_true = C_out + S_TRUE / Q_TRUE
    C_analytic = C_ss_true + (C0 - C_ss_true) * np.exp(-Q_TRUE * t_ref / V)
    ax.plot(t_ref, C_analytic, "--", color="#27ae60", lw=1.5, label="Analytical (true params)")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("CO₂ [ppm]")
    ax.set_title("PINN Prediction vs Measurements")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

def plot_loss_curves(ax, epochs_arr, history, warmup_epochs):
    ax.semilogy(epochs_arr, history["loss_total"], label="Total")
    ax.semilogy(epochs_arr, history["loss_data"], "--", label="Data")
    ax.semilogy(epochs_arr, history["loss_phys"], "--", label="Phys")
    ax.axvline(warmup_epochs, ls=":", c="gray")
    ax.set_title("Loss")
    ax.legend()
    ax.grid()


def plot_Q_recovery(ax, model, t_end, tau_true, Q_true):
    t = np.linspace(0, t_end, 500)

    # true
    Q_t_true = np.zeros_like(t)
    for i in range(len(Q_true)):
        if i == 0:
            mask = t < tau_true[0]
        elif i == len(Q_true) - 1:
            mask = t >= tau_true[-1]
        else:
            mask = (t >= tau_true[i-1]) & (t < tau_true[i])
        Q_t_true[mask] = Q_true[i]

    # estimated
    with torch.no_grad():
        T = torch.tensor(t.reshape(-1,1), dtype=torch.float32)
        Q_est, _, _ = build_piecewise(model, T, t.min(), t.max())
        Q_est = Q_est.numpy().flatten()

    ax.step(t, Q_t_true, label="True")
    ax.step(t, Q_est, "--", label="Estimated")
    ax.set_title("Q recovery")
    ax.legend()
    ax.grid()


def plot_S_recovery(ax, model, t_end, tau_true, S_true):
    t = np.linspace(0, t_end, 500)

    S_t_true = np.zeros_like(t)
    for i in range(len(S_true)):
        if i == 0:
            mask = t < tau_true[0]
        elif i == len(S_true) - 1:
            mask = t >= tau_true[-1]
        else:
            mask = (t >= tau_true[i-1]) & (t < tau_true[i])
        S_t_true[mask] = S_true[i]

    with torch.no_grad():
        T = torch.tensor(t.reshape(-1,1), dtype=torch.float32)
        _, S_est, _ = build_piecewise(model, T, t.min(), t.max())
        S_est = S_est.numpy().flatten()

    ax.step(t, S_t_true, label="True")
    ax.step(t, S_est, "--", label="Estimated")
    ax.set_title("S recovery")
    ax.legend()
    ax.grid()


def plot_concentration(ax, model, t_np, c_np, T_col, c_mean, c_std):

    with torch.no_grad():
        pred = model(T_col).numpy()

    ax.scatter(t_np, c_np, s=5, alpha=0.3)
    ax.plot(t_np, pred, lw=2)
    ax.set_title("C(t)")
    ax.grid()


# ============================================================
# Master plot (matches your earlier structure)
# ============================================================

def plot_all_diagnostics(model, history, t_np, c_np, T_col,
                         c_mean, c_std,
                         tau_true, Q_true, S_true):

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))

    epochs = np.arange(len(history["loss_total"]))

    plot_loss_curves(axes[0], epochs, history, 500)
    plot_concentration(axes[1], model, t_np, c_np, T_col, c_mean, c_std)
    plot_Q_recovery(axes[2], model, t_np.max(), tau_true, Q_true)
    plot_S_recovery(axes[3], model, t_np.max(), tau_true, S_true)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main(path, tau_true, Q_true, S_true, n_segments=16):

    t_np, c_np, t_train, c_train, t_test, c_test = load_data(path)

    t_min, t_max = t_np.min(), t_np.max()
    c_mean, c_std = c_train.mean(), c_train.std()

    T_train = torch.tensor(norm_t(t_train, t_min, t_max), dtype=torch.float32)
    C_train = torch.tensor(norm_c(c_train, c_mean, c_std), dtype=torch.float32)

    T_col = torch.tensor(
        norm_t(np.linspace(t_min, t_max, 1000).reshape(-1,1),
               t_min, t_max),
        dtype=torch.float32,
        requires_grad=True
    )

    model = PINN(n_segments)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = train(
        model, optimizer,
        T_train, C_train, T_col,
        t_min, t_max, c_mean, c_std
    )

    plot_all_diagnostics(model, history, t_np, c_np, T_col,
                         c_mean, c_std,
                         tau_true, Q_true, S_true)

    print("done")


if __name__ == "__main__":
    if len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("Usage: raapinn.py [path] [change_points_in_time] [true_Q's] [true_S's] [segment_number]")
