"""
Plotting functions for PINN diagnostics
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def plot_loss_curves(ax, epochs_arr, history, warmup_epochs):
    """Plot training loss curves (total, data, physics)."""
    ax.semilogy(epochs_arr, history["loss_total"], lw=1.5, label="Total loss", color="#2c3e50")
    ax.semilogy(epochs_arr, history["loss_data"],  lw=1.5, label="Data loss",  color="#e74c3c", ls="--")
    ax.semilogy(epochs_arr, history["loss_phys"],  lw=1.5, label="Phys loss",  color="#3498db", ls="--")
    ax.axvline(warmup_epochs, color="gray", ls=":", lw=1, label=f"Warm-up end ({warmup_epochs})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# def plot_Q_convergence(ax, epochs_arr, history, Q_TRUE):
#     """Plot Q parameter convergence."""
#     ax.plot(epochs_arr, history["Q"], lw=1.5, color="#e67e22")
#     # ax.axhline(Q_TRUE, color="black", ls="--", lw=1, label=f"True Q = {Q_TRUE:.2f}")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Q  [m³/h]")
#     ax.set_title("Recovered Q over Training")
#     ax.legend(fontsize=9)
#     ax.grid(alpha=0.3)


# def plot_S_convergence(ax, epochs_arr, history, S_TRUE):
#     """Plot S parameter convergence."""
#     ax.plot(epochs_arr, history["S"], lw=1.5, color="#8e44ad")
#     ax.axhline(S_TRUE, color="black", ls="--", lw=1, label=f"True S = {S_TRUE:.2e}")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("S  [ppm·m³/h]")
#     ax.set_title("Recovered S over Training")
#     ax.legend(fontsize=9)
#     ax.grid(alpha=0.3)


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
    # C_ss_true = C_out + S_TRUE / Q_TRUE
    # C_analytic = C_ss_true + (C0 - C_ss_true) * np.exp(-Q_TRUE * t_ref / V)
    # ax.plot(t_ref, C_analytic, "--", color="#27ae60", lw=1.5, label="Analytical (true params)")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("CO₂ [ppm]")
    ax.set_title("PINN Prediction vs Measurements")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_pde_residual(ax, model, t_col_np, physics_residual_fn, norm_t,
                      t_min, t_max, c_std, c_mean):
    """Plot PDE residual over collocation grid.
    
    FIX: norm_t needs t_min/t_max; physics_residual_fn needs model + extra args.
    """
    T_col_eval = torch.tensor(
        norm_t(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True
    )
    residual_eval = physics_residual_fn(model, T_col_eval, t_max, t_min, c_std, c_mean)
    res_np = residual_eval.detach().numpy().flatten()

    ax.plot(t_col_np.flatten(), res_np, lw=1, color="#c0392b", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(t_col_np.flatten(), res_np, alpha=0.15, color="#c0392b")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("PDE Residual (normalised)")
    ax.set_title("Physics Residual  V dC/dt − Q(C_out−C) − S")
    ax.grid(alpha=0.3)


def plot_all_diagnostics(model, history, t_np, c_np, t_train_np, c_train_np,
                         t_col_np, physics_residual_fn, norm_t,
                         epochs, warmup_epochs,
                         t_min, t_max, c_mean, c_std,
                         C_out, V, C0,
                         Q_TRUE=200.0, S_TRUE=1e5,
                         output_path="iaq_pinn_diagnostics.png"):
    """
    Generate all diagnostic plots for PINN training and save to output_path.

    output_path is used as-is (absolute or relative), so the caller
    (batch_train or main directly) controls which directory it lands in.
    """
    epochs_arr = np.arange(1, epochs + 1)

    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Loss curves — full width
    ax_loss = fig.add_subplot(gs[0, :])
    plot_loss_curves(ax_loss, epochs_arr, history, warmup_epochs)

    # Q and S convergence
    # ax_Q = fig.add_subplot(gs[1, 0])
    # plot_Q_convergence(ax_Q, epochs_arr, history, Q_TRUE)

    # ax_S = fig.add_subplot(gs[1, 1])
    # plot_S_convergence(ax_S, epochs_arr, history, S_TRUE)

    # Predictions vs measurements — full width
    ax_fit = fig.add_subplot(gs[2, :])
    plot_predictions_vs_measurements(
        ax_fit, model, t_np, c_np, t_train_np, c_train_np,
        t_min, t_max, c_mean, c_std, norm_t,
        C_out, S_TRUE, Q_TRUE, V, C0,
    )

    # PDE residual — full width; FIX: pass all required args
    ax_res = fig.add_subplot(gs[3, :])
    plot_pde_residual(
        ax_res, model, t_col_np, physics_residual_fn, norm_t,
        t_min, t_max, c_std, c_mean,
    )

    # Summary text box at the bottom
    Q_est = model.Q.item()
    S_est = model.S.item()
    textstr = (
        f"Recovered:  Q = {Q_est:.2f}   m³/h\n"
        f"            S = {S_est:.3e}    ppm·m³/h"
    )
    fig.text(0.5, 0.01, textstr, ha="center", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("PINN Diagnostics — IAQ CO₂ Parameter Recovery", fontsize=14, y=1.01)

    # FIX: save directly to output_path — caller is responsible for the directory
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)  # create dirs if needed
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure -> {out}")
    plt.close(fig)   # don't pop up a window during batch runs


# make functions for plottinsg varying Q or S with ....