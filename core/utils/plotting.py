from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from core.pinn.pinn_architecture import physics_residual
from core.utils.preprocessing import normalize_with_stats
from experiment.configs.schema import ExperimentConfig, TrueValues


# shared 

def plot_loss_curves(ax, epochs_arr, history, warmup_epochs):
    """Training loss curves — total, data, physics. Works for all three PINNs."""
    ax.semilogy(epochs_arr, history["loss_total"], lw=1.5,
                label="Total loss", color="#2c3e50")
    ax.semilogy(epochs_arr, history["loss_data"],  lw=1.5,
                label="Data loss",  color="#e74c3c", ls="--")
    ax.semilogy(epochs_arr, history["loss_phys"],  lw=1.5,
                label="Phys loss",  color="#3498db", ls="--")
    ax.axvline(warmup_epochs, color="gray", ls=":", lw=1,
               label=f"Warm-up end ({warmup_epochs})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_predictions(ax, model, t_np, c_np, t_train_np, c_train_np,
                     stats, show_analytical=False,
                     C_out=None, S_TRUE=None, Q_TRUE=None, V=None, C0=None):

    t_min  = stats["t_min"]
    t_max  = stats["t_max"]
    c_mean = stats["y_mean"]
    c_std  = stats["y_std"]

    t_full_np = np.linspace(t_np.min(), t_np.max(), 500).reshape(-1, 1).astype(np.float32)
    T_full    = torch.tensor(normalize_with_stats(t_full_np, t_min, t_max), dtype=torch.float32)

    with torch.no_grad():
        C_pred_phys = model(T_full).numpy() * c_std + c_mean

    ax.scatter(t_np,       c_np,       s=6,  alpha=0.3, color="#aaa",    label="C_meas (all)")
    ax.scatter(t_train_np, c_train_np, s=8,  alpha=0.5, color="#e74c3c", label="C_meas (train)")
    ax.plot(t_full_np, C_pred_phys, lw=2, color="#2c3e50", label="PINN prediction")

    if show_analytical and all(v is not None for v in [C_out, S_TRUE, Q_TRUE, V, C0]):
        t_ref      = t_full_np.flatten()
        C_ss_true  = C_out + S_TRUE / Q_TRUE
        C_analytic = C_ss_true + (C0 - C_ss_true) * np.exp(-Q_TRUE * t_ref / V)
        ax.plot(t_ref, C_analytic, "--", color="#27ae60", lw=1.5,
                label="Analytical (true params)")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("CO₂ [ppm]")
    ax.set_title("PINN Prediction vs Measurements")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_pde_residual(ax, model, t_col_np, stats, cfg, segment_boundaries=None):

    t_min  = stats["t_min"]
    t_max  = stats["t_max"]

    T_col_eval = torch.tensor(
        normalize_with_stats(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True
    )

    residual_eval = physics_residual(model, T_col_eval, stats, cfg.physics.V, cfg.physics.C_out)
    res_np        = residual_eval.detach().numpy().flatten()

    ax.plot(t_col_np.flatten(), res_np, lw=1, color="#c0392b", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(t_col_np.flatten(), res_np, alpha=0.15, color="#c0392b")

    if segment_boundaries is not None:
        for b in segment_boundaries:
            ax.axvline(b, color="gray", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("PDE Residual (normalised)")
    ax.set_title("Physics Residual  [should -> 0]")
    ax.grid(alpha=0.3)


# ── simple PINN specific ──────────────────────────────────────────────────────

def plot_scalar_convergence(ax, epochs_arr, history, key, true_val,
                            label, ylabel, color):

    values = np.array([
        np.asarray(v).reshape(-1)[0] for v in history[key]
    ])

    ax.plot(
        epochs_arr,
        values,
        lw=1.5,
        color=color,
        label=f"{label} estimated",
    )

    ax.axhline(true_val, color="black", ls="--", lw=1,
               label=f"True {label} = {true_val:.3g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Recovered {label} over Training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ── varying PINN specific 

def plot_mean_param_convergence(ax, epochs_arr, history, key, true_mean,
                                label, ylabel, color):

    values = np.array([np.asarray(v).reshape(-1) for v in history[key]])
    means = values.mean(axis=1)

    ax.plot(
        epochs_arr,
        means,
        lw=1.5,
        color=color,
        label=f"{label} mean (est)",
    )

    ax.axhline(
        true_mean,
        color="black",
        ls="--",
        lw=1,
        label=f"True {label} mean = {true_mean:.3g}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mean {label} over Training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_segment_bars(ax, true_seg, est_seg, label, ylabel,
                      true_color, est_color, scale=1.0):

    seg_ids = np.arange(len(true_seg))
    ax.bar(seg_ids - 0.2, true_seg * scale, width=0.4,
           color=true_color, alpha=0.7, label=f"{label} true")
    ax.bar(seg_ids + 0.2, est_seg  * scale, width=0.4,
           color=est_color,  alpha=0.7, label=f"{label} estimated")
    ax.set_xlabel("Segment index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-Segment {label}: True vs Estimated")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")


# ── RAA-PINN specific ─────────────────────────────────────────────────────────

def plot_stage1_scores(ax, t_np, scores, peak_indices=None):

    t_flat = t_np.flatten()
    ax.plot(t_flat, scores, lw=1.5, color="#2980b9", label="Window residual score")

    if peak_indices is not None and len(peak_indices) > 0:
        for idx in peak_indices:
            ax.axvline(t_flat[idx], color="#e74c3c", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(t_flat[peak_indices[0]], color="#e74c3c", ls="--",
                   lw=1.2, alpha=0.8, label="Detected changepoint")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual score")
    ax.set_title("Stage I — Sliding Window Physics Residual Score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_stage2_interval(ax, model, t_interval_np, c_interval_np,
                         tau_est, stats, interval_label=""):

    t_min  = stats["t_min"]
    t_max  = stats["t_max"]
    c_mean = stats["y_mean"]
    c_std  = stats["y_std"]

    t_fine = np.linspace(t_interval_np.min(), t_interval_np.max(),
                         200).reshape(-1, 1).astype(np.float32)
    T_fine = torch.tensor(normalize_with_stats(t_fine, t_min, t_max), dtype=torch.float32)

    with torch.no_grad():
        C_pred = model(T_fine).numpy() * c_std + c_mean

    ax.scatter(t_interval_np, c_interval_np, s=10, alpha=0.6,
               color="#888", label="C_meas")
    ax.plot(t_fine, C_pred, lw=2, color="#2c3e50", label="PINN fit")
    ax.axvline(tau_est, color="#e74c3c", ls="--", lw=1.5,
               label=f"τ̂ = {tau_est:.3f}h")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("CO₂ [ppm]")
    ax.set_title(f"Stage II fit{' — ' + interval_label if interval_label else ''}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

def plot_convergence(ax, epochs_arr, history, key, label, ylabel, color, true_val=None):
    """plots a single scalar value per epoch (Q for constant PINN, or one tau)"""
    values = np.array([np.asarray(v).reshape(-1)[0] for v in history[key]])
    ax.plot(epochs_arr, values, lw=1.5, color=color, label=f"{label} estimated")
    if true_val is not None:
        ax.axhline(true_val, color="black", ls="--", lw=1, label=f"True = {true_val:.3g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# Assemblers

def plot_all_simple(
    result:      dict,
    data:        dict,
    cfg:         ExperimentConfig,
    true_vals:   TrueValues | None = None,
    output_path: str = "simple_diagnostics.png",
):
    model     = result["model"]
    history   = result["history"]
    stats     = result["stats"]
    t_col_np  = result["t_col_np"]
    estimates = result["estimates"]

    t_np       = data["t_np"]
    c_np       = data["c_np"]
    t_train_np = data["t_train_np"]
    c_train_np = data["c_train_np"]

    Q_TRUE = true_vals.Q[0] if (true_vals and true_vals.Q is not None) else None
    S_TRUE = true_vals.S[0] if (true_vals and true_vals.S is not None) else None

    epochs_arr = np.arange(1, cfg.train.epochs + 1)

    fig = plt.figure(figsize=(14, 16))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :])
    plot_loss_curves(ax_loss, epochs_arr, history, cfg.train.warmup_epochs)

    ax_Q = fig.add_subplot(gs[1, 0])
    plot_convergence(ax_Q, epochs_arr, history,
                     key="Q", label="Q", ylabel="Q [m³/h]",
                     color="#e67e22", true_val=Q_TRUE)

    ax_S = fig.add_subplot(gs[1, 1])
    plot_convergence(ax_S, epochs_arr, history,
                     key="S", label="S", ylabel="S [ppm·m³/h]",
                     color="#8e44ad", true_val=S_TRUE)

    ax_fit = fig.add_subplot(gs[2, :])
    plot_predictions(ax_fit, model, t_np, c_np, t_train_np, c_train_np,
                     stats,
                     show_analytical=(true_vals is not None),
                     C_out=cfg.physics.C_out,
                     S_TRUE=S_TRUE, Q_TRUE=Q_TRUE,
                     V=cfg.physics.V, C0=cfg.physics.C0)

    ax_res = fig.add_subplot(gs[3, :])
    plot_pde_residual(ax_res, model, t_col_np, stats, cfg)

    Q_est = estimates["Q"][0]
    S_est = estimates["S"][0]
    textstr = (
        f"Recovered:  Q = {Q_est:.2f}" + (f"  (true {Q_TRUE:.0f})" if Q_TRUE else "") + "  m³/h\n"
        f"            S = {S_est:.3e}" + (f"  (true {S_TRUE:.2e})" if S_TRUE else "") + "  ppm·m³/h"
    )
    fig.text(0.5, 0.01, textstr, ha="center", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"{cfg.name} — Simple PINN", fontsize=13, y=1.01)
    _save(fig, output_path)


def plot_all_varying(
    result:      dict,
    data:        dict,
    cfg:         ExperimentConfig,
    true_vals:   TrueValues | None = None,
    output_path: str = "varying_diagnostics.png",
):
    model     = result["model"]
    history   = result["history"]
    stats     = result["stats"]
    t_col_np  = result["t_col_np"]
    estimates = result["estimates"]

    t_np       = data["t_np"]
    c_np       = data["c_np"]
    t_train_np = data["t_train_np"]
    c_train_np = data["c_train_np"]

    Q_est_seg = estimates["Q"]
    S_est_seg = estimates["S"]

    epochs_arr = np.arange(1, cfg.train.epochs + 1)
    boundaries = np.arange(1, cfg.n_segments) * cfg.segment_duration

    fig = plt.figure(figsize=(15, 20))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :])
    plot_loss_curves(ax_loss, epochs_arr, history, cfg.train.warmup_epochs)

    Q_true_mean = np.mean(true_vals.Q) if (true_vals and true_vals.Q is not None) else None
    S_true_mean = np.mean(true_vals.S) if (true_vals and true_vals.S is not None) else None

    ax_Qmean = fig.add_subplot(gs[1, 0])
    plot_mean_param_convergence(ax_Qmean, epochs_arr, history,
                                key="Q", true_mean=Q_true_mean,
                                label="Q", ylabel="Q [m³/h]", color="#e67e22")

    ax_Smean = fig.add_subplot(gs[1, 1])
    plot_mean_param_convergence(ax_Smean, epochs_arr, history,
                                key="S", true_mean=S_true_mean,
                                label="S", ylabel="S [ppm·m³/h]", color="#8e44ad")

    ax_Qbar = fig.add_subplot(gs[2, 0])
    plot_segment_bars(ax_Qbar,
                      true_seg=np.array(true_vals.Q) if (true_vals and true_vals.Q is not None) else np.zeros_like(Q_est_seg),
                      est_seg=Q_est_seg,
                      label="Q", ylabel="Q [m³/h]",
                      true_color="#2980b9", est_color="#e67e22")

    ax_Sbar = fig.add_subplot(gs[2, 1])
    plot_segment_bars(ax_Sbar,
                      true_seg=np.array(true_vals.S) if (true_vals and true_vals.S is not None) else np.zeros_like(S_est_seg),
                      est_seg=S_est_seg,
                      label="S_vol", ylabel="S_vol [m³ CO₂/h]",
                      true_color="#27ae60", est_color="#8e44ad", scale=1/1e6)

    ax_fit = fig.add_subplot(gs[3, :])
    plot_predictions(ax_fit, model, t_np, c_np, t_train_np, c_train_np,
                     stats, show_analytical=False)

    ax_res = fig.add_subplot(gs[4, :])
    plot_pde_residual(ax_res, model, t_col_np, stats, cfg,
                      segment_boundaries=boundaries)

    fig.suptitle(f"{cfg.name} — Varying PINN", fontsize=13, y=1.01)
    _save(fig, output_path)


def plot_all_raa(
    stage1_result:  dict,
    stage2_results: list[dict],
    data:           dict,
    cfg:            ExperimentConfig,
    true_vals:      TrueValues | None = None,
    output_path:    str = "raa_diagnostics.png",
):
    t_np = data["t_np"]
    c_np = data["c_np"]

    # extract changepoint times and Q/S from estimates
    results_sorted    = sorted(stage2_results, key=lambda r: r["estimates"]["taus"][0])
    changepoint_times = [r["estimates"]["taus"][0] for r in results_sorted]
    Q_values          = [results_sorted[0]["estimates"]["Q"][0]] + [r["estimates"]["Q"][1] for r in results_sorted]
    S_values          = [results_sorted[0]["estimates"]["S"][0]] + [r["estimates"]["S"][1] for r in results_sorted]

    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_scores = fig.add_subplot(gs[0, :])
    plot_stage1_scores(ax_scores, stage1_result["t_np"],
                       stage1_result["scores"], stage1_result["peak_indices"])

    ax_Q = fig.add_subplot(gs[1, :])
    ax_S = fig.add_subplot(gs[2, :])
    plot_piecewise_params(
        ax_Q, ax_S, t_np,
        Q_true_np=np.array(true_vals.Q) if (true_vals and true_vals.Q is not None) else np.zeros(len(t_np)),
        S_true_np=np.array(true_vals.S) if (true_vals and true_vals.S is not None) else np.zeros(len(t_np)),
        changepoint_times=changepoint_times,
        Q_values=Q_values,
        S_values=S_values,
    )

    fig.suptitle(f"{cfg.name} — RAA-PINN", fontsize=13, y=1.01)
    _save(fig, output_path)


def plot_all_raa_training(
    stage2_results: list[dict],
    cfg:            ExperimentConfig,
    output_path:    str = "raa_training_diagnostics.png",
):
    if not stage2_results:
        return

    n          = len(stage2_results)
    fig        = plt.figure(figsize=(14, 4 * n))
    gs         = gridspec.GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.35)

    for i, result in enumerate(stage2_results):
        history    = result["history"]
        epochs_arr = np.arange(1, cfg.train.epochs + 1)
        label      = f"Changepoint {i + 1}"

        ax_loss = fig.add_subplot(gs[i, 0])
        plot_loss_curves(ax_loss, epochs_arr, history, cfg.train.warmup_epochs)
        ax_loss.set_title(f"Losses — {label}")

        ax_Q = fig.add_subplot(gs[i, 1])
        plot_param_convergence(ax_Q, epochs_arr, history,
                               key="Q", ylabel="Q [m³/h]", title=f"Q — {label}")

        ax_S = fig.add_subplot(gs[i, 2])
        plot_param_convergence(ax_S, epochs_arr, history,
                               key="S", ylabel="S [ppm·m³/h]", title=f"S — {label}")

    fig.suptitle(f"{cfg.name} — RAA-PINN Stage II Training", fontsize=13, y=1.01)
    _save(fig, output_path)