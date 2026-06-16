from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


# ── shared ────────────────────────────────────────────────────────────────────

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
                     stats, norm_t, show_analytical=False,
                     C_out=None, S_TRUE=None, Q_TRUE=None, V=None, C0=None):
    """
    Predicted C(t) vs measurements. Works for all three PINNs.

    show_analytical: only pass True for the simple constant PINN where
                     a single exponential analytical solution exists.
                     Requires C_out, S_TRUE, Q_TRUE, V, C0 to be passed.
    """
    t_min  = stats["t_min"]
    t_max  = stats["t_max"]
    c_mean = stats["y_mean"]
    c_std  = stats["y_std"]

    t_full_np = np.linspace(t_np.min(), t_np.max(), 500).reshape(-1, 1).astype(np.float32)
    T_full    = torch.tensor(norm_t(t_full_np, t_min, t_max), dtype=torch.float32)

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


def plot_pde_residual(ax, model, t_col_np, physics_residual_fn, stats, norm_t,
                      segment_boundaries=None):
    """
    PDE residual over collocation grid. Works for all three PINNs.

    segment_boundaries: optional list of times to draw vertical grey lines,
                        used by varying PINN to mark known segment edges.
    """
    t_min  = stats["t_min"]
    t_max  = stats["t_max"]
    c_mean = stats["y_mean"]
    c_std  = stats["y_std"]

    T_col_eval = torch.tensor(
        norm_t(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True
    )
    residual_eval = physics_residual_fn(model, T_col_eval, stats)
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
    """
    Single scalar parameter convergence over epochs.
    key: history dict key, e.g. "Q" or "S"
    """
    ax.plot(epochs_arr, history[key], lw=1.5, color=color, label=f"{label} estimated")
    ax.axhline(true_val, color="black", ls="--", lw=1,
               label=f"True {label} = {true_val:.3g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Recovered {label} over Training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


# ── varying PINN specific ─────────────────────────────────────────────────────

def plot_mean_param_convergence(ax, epochs_arr, history, key, true_mean,
                                label, ylabel, color):
    """
    Mean of segment parameters over epochs.
    key: history dict key, e.g. "Q_mean" or "S_mean"
    """
    ax.plot(epochs_arr, history[key], lw=1.5, color=color, label=f"{label} mean (est)")
    ax.axhline(true_mean, color="black", ls="--", lw=1,
               label=f"True {label} mean = {true_mean:.3g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mean {label} over Training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_segment_bars(ax, true_seg, est_seg, label, ylabel,
                      true_color, est_color, scale=1.0):
    """
    Side-by-side bar chart of true vs estimated values per segment.

    scale: multiply both arrays before plotting, e.g. 1/1e6 to show S_vol
    """
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
    """
    Stage I sliding window residual scores over time.
    Peak locations are marked with vertical lines.
    """
    t_flat = t_np.flatten()
    ax.plot(t_flat, scores, lw=1.5, color="#2980b9", label="Window residual score")

    if peak_indices is not None and len(peak_indices) > 0:
        for idx in peak_indices:
            ax.axvline(t_flat[idx], color="#e74c3c", ls="--", lw=1.2, alpha=0.8)
        # add one dummy line for legend
        ax.axvline(t_flat[peak_indices[0]], color="#e74c3c", ls="--",
                   lw=1.2, alpha=0.8, label="Detected changepoint")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual score")
    ax.set_title("Stage I — Sliding Window Physics Residual Score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_stage2_interval(ax, model, t_interval_np, c_interval_np,
                         tau_est, stats, norm_t, interval_label=""):
    """
    C(t) prediction on one candidate interval with vertical line at estimated tau.
    Used once per detected changepoint.
    """
    t_min  = stats["t_min"]
    t_max  = stats["t_max"]
    c_mean = stats["y_mean"]
    c_std  = stats["y_std"]

    t_fine = np.linspace(t_interval_np.min(), t_interval_np.max(),
                         200).reshape(-1, 1).astype(np.float32)
    T_fine = torch.tensor(norm_t(t_fine, t_min, t_max), dtype=torch.float32)

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


def plot_piecewise_params(ax_Q, ax_S, t_np,
                          Q_true_np, S_true_np,
                          changepoint_times, Q_values, S_values):
    """
    Full piecewise Q(t) and S(t) step functions over the whole time range.

    changepoint_times: sorted list of estimated tau values, e.g. [1.5, 3.2, 5.1]
    Q_values: list of Q estimates for each segment — length = len(changepoints) + 1
    S_values: list of S estimates for each segment — length = len(changepoints) + 1
    """
    t_flat = t_np.flatten()

    # build estimated piecewise arrays
    Q_est_arr = np.zeros_like(t_flat)
    S_est_arr = np.zeros_like(t_flat)

    boundaries = [t_flat[0]] + list(changepoint_times) + [t_flat[-1]]
    for i in range(len(Q_values)):
        mask = (t_flat >= boundaries[i]) & (t_flat < boundaries[i + 1])
        Q_est_arr[mask] = Q_values[i]
        S_est_arr[mask] = S_values[i]

    # Q plot
    ax_Q.step(t_flat, Q_true_np, where="post", lw=2,
              color="#2980b9", label="Q true")
    ax_Q.step(t_flat, Q_est_arr, where="post", lw=2,
              color="#e67e22", ls="--", label="Q estimated")
    for tau in changepoint_times:
        ax_Q.axvline(tau, color="#e74c3c", ls=":", lw=0.8, alpha=0.6)
    ax_Q.set_xlabel("Time [h]")
    ax_Q.set_ylabel("Q [m³/h]")
    ax_Q.set_title("Estimated vs True Q(t)")
    ax_Q.legend(fontsize=9)
    ax_Q.grid(alpha=0.3)

    # S plot
    ax_S.step(t_flat, S_true_np / 1e6, where="post", lw=2,
              color="#27ae60", label="S_vol true")
    ax_S.step(t_flat, S_est_arr / 1e6, where="post", lw=2,
              color="#8e44ad", ls="--", label="S_vol estimated")
    for tau in changepoint_times:
        ax_S.axvline(tau, color="#e74c3c", ls=":", lw=0.8, alpha=0.6)
    ax_S.set_xlabel("Time [h]")
    ax_S.set_ylabel("S_vol [m³ CO₂/h]")
    ax_S.set_title("Estimated vs True S(t)")
    ax_S.legend(fontsize=9)
    ax_S.grid(alpha=0.3)


# ── assemblers ────────────────────────────────────────────────────────────────

def plot_all_simple(model, history, t_np, c_np, t_train_np, c_train_np,
                    t_col_np, physics_residual_fn, stats, norm_t,
                    epochs, warmup_epochs,
                    C_out, V, C0, Q_TRUE, S_TRUE,
                    output_path="iaq_pinn_simple_diagnostics.png"):
    """
    Diagnostic plot for the simple constant PINN.
    Panels: losses | Q convergence | S convergence | prediction | residual
    """
    epochs_arr = np.arange(1, epochs + 1)

    fig = plt.figure(figsize=(14, 16))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :])
    plot_loss_curves(ax_loss, epochs_arr, history, warmup_epochs)

    ax_Q = fig.add_subplot(gs[1, 0])
    plot_scalar_convergence(ax_Q, epochs_arr, history,
                            key="Q", true_val=Q_TRUE,
                            label="Q", ylabel="Q [m³/h]", color="#e67e22")

    ax_S = fig.add_subplot(gs[1, 1])
    plot_scalar_convergence(ax_S, epochs_arr, history,
                            key="S", true_val=S_TRUE,
                            label="S", ylabel="S [ppm·m³/h]", color="#8e44ad")

    ax_fit = fig.add_subplot(gs[2, :])
    plot_predictions(ax_fit, model, t_np, c_np, t_train_np, c_train_np,
                     stats, norm_t,
                     show_analytical=True,
                     C_out=C_out, S_TRUE=S_TRUE, Q_TRUE=Q_TRUE, V=V, C0=C0)

    ax_res = fig.add_subplot(gs[3, :])
    plot_pde_residual(ax_res, model, t_col_np, physics_residual_fn, stats, norm_t)

    Q_est = model.param_model.get_Q_S(
        torch.tensor([[0.0]])
    )[0].item()
    S_est = model.param_model.get_Q_S(
        torch.tensor([[0.0]])
    )[1].item()
    textstr = (
        f"Recovered:  Q = {Q_est:.2f}  (true {Q_TRUE:.0f})  m³/h\n"
        f"            S = {S_est:.3e}  (true {S_TRUE:.2e})  ppm·m³/h"
    )
    fig.text(0.5, 0.01, textstr, ha="center", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Simple PINN — Constant Q and S Recovery", fontsize=13, y=1.01)
    _save(fig, output_path)


def plot_all_varying(model, history, t_np, c_np, t_train_np, c_train_np,
                     t_col_np, physics_residual_fn, stats, norm_t,
                     Q_true_np, S_true_np,
                     n_segments, segment_duration,
                     epochs, warmup_epochs,
                     output_path="iaq_pinn_varying_diagnostics.png"):
    """
    Diagnostic plot for the varying piecewise PINN.
    Panels: losses | mean Q/S convergence | segment bars | prediction | residual
    """
    epochs_arr  = np.arange(1, epochs + 1)
    t_hours_all = np.linspace(float(t_np.min()), float(t_np.max()), len(Q_true_np))
    seg_centers = (np.arange(n_segments) + 0.5) * segment_duration

    Q_true_seg = np.array([
        Q_true_np[np.argmin(np.abs(t_hours_all - tc))] for tc in seg_centers
    ])
    S_true_seg = np.array([
        S_true_np[np.argmin(np.abs(t_hours_all - tc))] for tc in seg_centers
    ])

    Q_est_seg = model.param_model.log_Q_segments.exp().detach().numpy() \
        if hasattr(model.param_model, "log_Q_segments") \
        else np.full(n_segments, model.param_model.log_Q.exp().item())

    S_est_seg = model.param_model.log_S_segments.exp().detach().numpy() \
        if hasattr(model.param_model, "log_S_segments") \
        else np.full(n_segments, model.param_model.log_S.exp().item())

    boundaries = np.arange(1, n_segments) * segment_duration

    fig = plt.figure(figsize=(15, 20))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :])
    plot_loss_curves(ax_loss, epochs_arr, history, warmup_epochs)

    ax_Qmean = fig.add_subplot(gs[1, 0])
    plot_mean_param_convergence(ax_Qmean, epochs_arr, history,
                                key="Q_mean", true_mean=Q_true_seg.mean(),
                                label="Q", ylabel="Q [m³/h]", color="#e67e22")

    ax_Smean = fig.add_subplot(gs[1, 1])
    plot_mean_param_convergence(ax_Smean, epochs_arr, history,
                                key="S_mean", true_mean=S_true_seg.mean(),
                                label="S", ylabel="S [ppm·m³/h]", color="#8e44ad")

    ax_Qbar = fig.add_subplot(gs[2, 0])
    plot_segment_bars(ax_Qbar, Q_true_seg, Q_est_seg,
                      label="Q", ylabel="Q [m³/h]",
                      true_color="#2980b9", est_color="#e67e22")

    ax_Sbar = fig.add_subplot(gs[2, 1])
    plot_segment_bars(ax_Sbar, S_true_seg, S_est_seg,
                      label="S_vol", ylabel="S_vol [m³ CO₂/h]",
                      true_color="#27ae60", est_color="#8e44ad", scale=1/1e6)

    ax_fit = fig.add_subplot(gs[3, :])
    plot_predictions(ax_fit, model, t_np, c_np, t_train_np, c_train_np,
                     stats, norm_t, show_analytical=False)

    ax_res = fig.add_subplot(gs[4, :])
    plot_pde_residual(ax_res, model, t_col_np, physics_residual_fn, stats, norm_t,
                      segment_boundaries=boundaries)

    fig.suptitle("Varying PINN — Piecewise Q and S Recovery", fontsize=13, y=1.01)
    _save(fig, output_path)


def plot_all_raa(t_np, c_np, scores, peak_indices,
                 stage2_results, Q_true_np, S_true_np,
                 output_path="iaq_pinn_raa_diagnostics.png"):
    """
    Diagnostic plot for RAA-PINN.
    Panels: stage1 scores | piecewise Q(t) | piecewise S(t)

    stage2_results: list of dicts, one per detected changepoint, each containing:
        {
            "tau":     float,   estimated changepoint time
            "Q_minus": float,   Q before the jump
            "Q_plus":  float,   Q after the jump
            "S_minus": float,   S before the jump
            "S_plus":  float,   S after the jump
        }
    """
    # extract changepoint times and build segment Q/S arrays
    changepoint_times = sorted([r["tau"] for r in stage2_results])

    # Q values: first segment gets Q_minus of first changepoint,
    # then each Q_plus becomes the next segment's value
    results_sorted = sorted(stage2_results, key=lambda r: r["tau"])
    Q_values = [results_sorted[0]["Q_minus"]] + [r["Q_plus"] for r in results_sorted]
    S_values = [results_sorted[0]["S_minus"]] + [r["S_plus"] for r in results_sorted]

    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # stage 1 scores full width
    ax_scores = fig.add_subplot(gs[0, :])
    plot_stage1_scores(ax_scores, t_np, scores, peak_indices)

    # piecewise Q and S
    ax_Q = fig.add_subplot(gs[1, :])
    ax_S = fig.add_subplot(gs[2, :])
    plot_piecewise_params(ax_Q, ax_S, t_np,
                          Q_true_np, S_true_np,
                          changepoint_times, Q_values, S_values)

    fig.suptitle("RAA-PINN — Unknown Boundary Parameter Recovery", fontsize=13, y=1.01)
    _save(fig, output_path)


# --- internal helper ---

def _save(fig, output_path):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)