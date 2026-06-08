import numpy as np
import matplotlib.pyplot as plt
import torch


# ============================================================
# Loss Curves
# ============================================================

def plot_loss_curves(ax, epochs_arr, history, warmup_epochs):
    """Plot training loss curves."""

    ax.semilogy(
        epochs_arr,
        history["loss_total"],
        lw=1.5,
        color="#2c3e50",
        label="Total loss"
    )

    ax.semilogy(
        epochs_arr,
        history["loss_data"],
        lw=1.5,
        ls="--",
        color="#e74c3c",
        label="Data loss"
    )

    ax.semilogy(
        epochs_arr,
        history["loss_phys"],
        lw=1.5,
        ls="--",
        color="#3498db",
        label="Physics loss"
    )

    ax.axvline(
        warmup_epochs,
        color="gray",
        ls=":",
        lw=1,
        label=f"Warm-up end ({warmup_epochs})"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ============================================================
# Concentration Recovery
# ============================================================

def plot_concentration_recovery(
    ax,
    t_true,
    C_true,
    t_meas,
    C_meas,
    t_pred,
    C_pred,
):
    """
    Plot true concentration, noisy measurements,
    and PINN prediction.
    """

    ax.plot(
        t_true,
        C_true,
        lw=2,
        color="#27ae60",
        label="True concentration"
    )

    ax.plot(
        t_pred,
        C_pred,
        lw=2,
        color="#2980b9",
        label="PINN prediction"
    )

    ax.scatter(
        t_meas,
        C_meas,
        s=18,
        color="#e74c3c",
        alpha=0.7,
        label="Measurements"
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("CO₂ concentration [ppm]")
    ax.set_title("Concentration Recovery")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ============================================================
# Piecewise Reconstruction
# ============================================================

def piecewise_schedule(t, tau, values):
    """
    Reconstruct piecewise-constant schedule.

    Parameters
    ----------
    t : array-like

    tau : list/array
        Changepoints.
        Example:
        [1.2, 3.7, 6.1]

    values : list/array
        Segment values.
        Example:
        [200, 500, 100, 700]
    """

    t = np.asarray(t)

    y = np.zeros_like(t, dtype=float)

    for i in range(len(values)):

        if i == 0:
            mask = t < tau[0]

        elif i == len(values) - 1:
            mask = t >= tau[-1]

        else:
            mask = (t >= tau[i - 1]) & (t < tau[i])

        y[mask] = values[i]

    return y


# ============================================================
# Q Recovery
# ============================================================

def plot_Q_recovery(
    ax,
    t_end,
    tau_true,
    Q_true,
    tau_hat,
    Q_hat,
):
    """
    Compare true and recovered Q(t).

    tau_true : changepoints
    Q_true   : segment values

    tau_hat  : recovered changepoints
    Q_hat    : recovered segment values
    """

    t_plot = np.linspace(0, t_end, 2000)

    Q_true_plot = piecewise_schedule(
        t_plot,
        tau_true,
        Q_true
    )

    Q_hat_plot = piecewise_schedule(
        t_plot,
        tau_hat,
        Q_hat
    )

    ax.step(
        t_plot,
        Q_true_plot,
        where="post",
        lw=2,
        color="#27ae60",
        label="True Q"
    )

    ax.step(
        t_plot,
        Q_hat_plot,
        where="post",
        lw=2,
        ls="--",
        color="#2980b9",
        label="Recovered Q"
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Q [m³/h]")
    ax.set_title("Q Recovery")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ============================================================
# S (or L) Recovery
# ============================================================

def plot_S_recovery(
    ax,
    t_end,
    tau_true,
    S_true,
    tau_hat,
    S_hat,
    ylabel="S [ppm·m³/h]"
):
    """
    Compare true and recovered S(t) or L(t).
    """

    t_plot = np.linspace(0, t_end, 2000)

    S_true_plot = piecewise_schedule(
        t_plot,
        tau_true,
        S_true
    )

    S_hat_plot = piecewise_schedule(
        t_plot,
        tau_hat,
        S_hat
    )

    ax.step(
        t_plot,
        S_true_plot,
        where="post",
        lw=2,
        color="#27ae60",
        label="True"
    )

    ax.step(
        t_plot,
        S_hat_plot,
        where="post",
        lw=2,
        ls="--",
        color="#2980b9",
        label="Recovered"
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel(ylabel)
    ax.set_title("Source Recovery")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ============================================================
# PDE Residual
# ============================================================

def plot_pde_residual(
    ax,
    model,
    t_col_np,
    physics_residual_fn,
    norm_t,
    t_min,
    t_max,
    c_std,
    c_mean,
):
    """
    Plot PDE residual over collocation grid.
    """

    T_col_eval = torch.tensor(
        norm_t(t_col_np, t_min, t_max),
        dtype=torch.float32,
        requires_grad=True,
    )

    residual_eval = physics_residual_fn(
        model,
        T_col_eval,
        t_max,
        t_min,
        c_std,
        c_mean,
    )

    res_np = residual_eval.detach().numpy().flatten()

    ax.plot(
        t_col_np.flatten(),
        res_np,
        lw=1,
        color="#c0392b",
        alpha=0.8,
    )

    ax.axhline(
        0,
        color="black",
        lw=0.8,
        ls="--",
    )

    ax.fill_between(
        t_col_np.flatten(),
        res_np,
        alpha=0.15,
        color="#c0392b",
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual")
    ax.set_title("Physics Residual")
    ax.grid(alpha=0.3)


# ============================================================
# Master Plot
# ============================================================

def plot_all_diagnostics(
    epochs_arr,
    history,
    warmup_epochs,
    t_true,
    C_true,
    t_meas,
    C_meas,
    t_pred,
    C_pred,
    t_end,
    tau_true,
    Q_true,
    tau_hat,
    Q_hat,
    S_true,
    S_hat,
):
    """
    Create a 4-panel diagnostic figure.
    """

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 16),
        constrained_layout=True
    )

    plot_loss_curves(
        axes[0],
        epochs_arr,
        history,
        warmup_epochs,
    )

    plot_concentration_recovery(
        axes[1],
        t_true,
        C_true,
        t_meas,
        C_meas,
        t_pred,
        C_pred,
    )

    plot_Q_recovery(
        axes[2],
        t_end,
        tau_true,
        Q_true,
        tau_hat,
        Q_hat,
    )

    plot_S_recovery(
        axes[3],
        t_end,
        tau_true,
        S_true,
        tau_hat,
        S_hat,
    )

    return fig, axes