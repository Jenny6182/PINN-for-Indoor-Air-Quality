from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.pinn.pinn_architecture import physics_residual
from core.utils.preprocessing import normalize_with_stats
from experiment.configs.schema import ExperimentConfig, TrueValues

# ---- core helpers ----
def style_ax(ax, title=None, xlabel=None, ylabel=None, legend=True):
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend: ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def get_epochs(cfg):
    return np.arange(1, cfg.train.epochs + 1)


def history_to_array(x):
    return np.array([np.asarray(v).reshape(-1) for v in x])


def safe_np(x, default):
    return np.array(x) if x is not None else default


# ---- loss plot ----
def plot_losses(ax, epochs, history, warmup_epochs):

    ax.semilogy(epochs, history["loss_total"], lw=1.5,
                label="Total loss", color="#2c3e50")

    ax.semilogy(epochs, history["loss_data"], lw=1.5,
                label="Data loss", color="#e74c3c", ls="--")

    ax.semilogy(epochs, history["loss_phys"], lw=1.5,
                label="Phys loss", color="#3498db", ls="--")

    ax.axvline(warmup_epochs, color="gray", ls=":",
               lw=1, label=f"Warm-up end ({warmup_epochs})")

    style_ax(ax, "Training Losses", "Epoch", "Loss")


# ---- unified PINN plotting functions ----

def plot_data_and_model(ax, model, t_np, c_np, t_train_np, c_train_np, stats):
    t_min, t_max = stats["t_min"], stats["t_max"]
    c_mean, c_std = stats["y_mean"], stats["y_std"]

    t_full = np.linspace(t_np.min(), t_np.max(), 500).reshape(-1, 1).astype(np.float32)
    T = torch.tensor(normalize_with_stats(t_full, t_min, t_max), dtype=torch.float32)

    with torch.no_grad():
        pred = model(T).numpy() * c_std + c_mean

    ax.scatter(t_np, c_np, s=6, alpha=0.3, color="green", label="data")
    ax.scatter(t_train_np, c_train_np, s=8, alpha=0.5, color="#e74c3c", label="train")
    ax.plot(t_full, pred, lw=2, color="#2c3e50", label="model")


# ---- history plotting ----
def plot_history(ax, epochs, history, key, mode="segments",
                 color=None, label=None, true=None):

    arr = history_to_array(history[key])

    if mode == "segments":
        for i in range(arr.shape[1]):
            ax.plot(epochs, arr[:, i], lw=1.5, label=f"{key}{i+1}")

    elif mode == "mean":
        mean = arr.mean(axis=1)
        ax.plot(epochs, mean, lw=1.5, color=color,
                label=f"{label} mean")

        if true is not None:
            ax.axhline(true, color="black", ls="--", lw=1, label="true")

    style_ax(ax, f"{key} training", "Epoch", key)


# ---- physics residual ----
def plot_pde_residual(ax, model, t_col_np, stats, cfg, segment_boundaries=None):
    t_min, t_max = stats["t_min"], stats["t_max"]

    t_col_np = np.asarray(t_col_np).reshape(-1, 1)
    t_phys = t_col_np * (t_max - t_min) + t_min

    T = torch.tensor(
        normalize_with_stats(t_col_np, t_min, t_max),
        dtype=torch.float32,
        requires_grad=True
    )

    res = physics_residual(model, T, stats, cfg.physics.V, cfg.physics.C_out)
    res = res.detach().numpy().flatten()

    ax.plot(t_phys.flatten(), res, lw=1, color="#c0392b", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(t_phys.flatten(), res, alpha=0.15, color="#c0392b")

    if segment_boundaries is not None:
        for b in np.atleast_1d(segment_boundaries):
            ax.axvline(b, color="gray", ls=":", lw=1.0, alpha=0.8)

    style_ax(ax, "Physics Residual", "Time [h]", "Residual")


# ---- RAA helpers ----
def plot_stage1(ax, t_np, scores, peaks):
    ax.plot(t_np.flatten(), scores, lw=1.5, color="#2980b9")

    if peaks is not None:
        for i in np.asarray(peaks).tolist():
            ax.axvline(t_np.flatten()[i], color="#e74c3c", ls="--", lw=1.2)

    style_ax(ax, "Stage I residual score", "Time", "Score")


def plot_piecewise(ax_Q, ax_S, t_np, Q_true, S_true, taus, Q_vals, S_vals):
    t = t_np.flatten()

    Q_est = np.zeros_like(t)
    S_est = np.zeros_like(t)

    taus = np.atleast_1d(taus)

    bounds = [t[0]] + list(taus) + [t[-1]]

    for i in range(len(Q_vals)):
        m = (t >= bounds[i]) & (t < bounds[i + 1])
        Q_est[m] = Q_vals[i]
        S_est[m] = S_vals[i]

    # Q
    ax_Q.step(t, Q_true, where="post", lw=2, label="true")
    ax_Q.step(t, Q_est, where="post", lw=2, ls="--", label="est")

    for tau in taus:
        ax_Q.axvline(tau, color="purple", ls=":", lw=1.0, alpha=1)

    style_ax(ax_Q, "Q(t)", "Time", "Q")

    # S
    ax_S.step(t, S_true / 1e6, where="post", lw=2, label="true")
    ax_S.step(t, S_est / 1e6, where="post", lw=2, ls="--", label="est")

    for tau in taus:
        ax_S.axvline(tau, color="purple", ls=":", lw=1.0, alpha=1)

    style_ax(ax_S, "S(t)", "Time", "S")


# ---- assemblers ----
def plot_all_raa(stage1_result, stage2_results, data, cfg, true_vals=None, output_path="raa.png"):

    result = stage2_results[0]

    model = result["model"]
    history = result["history"]
    stats = result["stats"]
    t_col = result["t_col_np"]
    est = result["estimates"]

    t_np = data["t_np"]
    c_np = data["c_np"]
    t_tr = data["t_train_np"]
    c_tr = data["c_train_np"]

    fig, axes = plt.subplots(6, 1, figsize=(15, 24), constrained_layout=True)

    ax_loss, ax_s1, ax_Q, ax_S, ax_pred, ax_res = axes

    epochs = get_epochs(cfg)

    # FIXED: full loss plot restored
    plot_losses(ax_loss, epochs, history, cfg.train.warmup_epochs)

    plot_stage1(ax_s1,
                stage1_result["t_np"],
                stage1_result["scores"],
                stage1_result["peak_indices"])

    plot_piecewise(
        ax_Q, ax_S,
        t_np,
        safe_np(true_vals.Q, np.zeros(len(t_np))),
        safe_np(true_vals.S, np.zeros(len(t_np))),
        est["taus"] if est["taus"] is not None else [],
        est["Q"], est["S"]
    )

    plot_data_and_model(ax_pred, model, t_np, c_np, t_tr, c_tr, stats)

    plot_pde_residual(ax_res, model, t_col, stats, cfg,
                      segment_boundaries=est["taus"])

    fig.suptitle(cfg.name, fontsize=13)
    _save(fig, output_path)


def plot_all_raa_training(stage2_results, cfg, output_path="raa_training.png"):

    if not stage2_results:
        return

    history = stage2_results[0]["history"]
    epochs = get_epochs(cfg)

    Q = history_to_array(history["Q"])
    S = history_to_array(history["S"])

    n = Q.shape[1] if Q.ndim > 1 else 1

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), constrained_layout=True)

    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        ax_Q, ax_S = axes[i]

        ax_Q.plot(epochs, Q[:, i], lw=1.5)
        style_ax(ax_Q, f"Q seg {i+1}", "Epoch", "Q")

        ax_S.plot(epochs, S[:, i], lw=1.5)
        style_ax(ax_S, f"S seg {i+1}", "Epoch", "S")

    fig.suptitle(cfg.name)
    _save(fig, output_path)


def _save(fig, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_all_simple(result, data, cfg, true_vals=None, output_path="simple.png"):

    model = result["model"]
    history = result["history"]
    stats = result["stats"]

    t_np = data["t_np"]
    c_np = data["c_np"]
    t_tr = data["t_train_np"]
    c_tr = data["c_train_np"]

    epochs = get_epochs(cfg)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), constrained_layout=True)
    ax_loss, ax_Q, ax_S, ax_pred = axes

    plot_losses(ax_loss, epochs, history, cfg.train.warmup_epochs)

    plot_history(ax_Q, epochs, history, "Q", mode="mean",
                 color="#e67e22", label="Q",
                 true=np.mean(true_vals.Q) if true_vals else None)

    plot_history(ax_S, epochs, history, "S", mode="mean",
                 color="#8e44ad", label="S",
                 true=np.mean(true_vals.S) if true_vals else None)

    plot_data_and_model(ax_pred, model, t_np, c_np, t_tr, c_tr, stats)

    fig.suptitle(f"{cfg.name} — Simple PINN", fontsize=13)
    _save(fig, output_path)


def plot_all_varying(result, data, cfg, true_vals=None, output_path="varying.png"):

    model = result["model"]
    history = result["history"]
    stats = result["stats"]
    t_col = result["t_col_np"]
    est = result["estimates"]

    t_np = data["t_np"]
    c_np = data["c_np"]
    t_tr = data["t_train_np"]
    c_tr = data["t_train_np"]

    epochs = get_epochs(cfg)

    fig, axes = plt.subplots(5, 1, figsize=(15, 20), constrained_layout=True)
    ax_loss, ax_Q, ax_S, ax_pred, ax_res = axes

    plot_losses(ax_loss, epochs, history, cfg.train.warmup_epochs)

    plot_history(ax_Q, epochs, history, "Q", mode="segments")
    plot_history(ax_S, epochs, history, "S", mode="segments")

    plot_data_and_model(ax_pred, model, t_np, c_np, t_tr, c_tr, stats)

    plot_pde_residual(ax_res, model, t_col, stats, cfg,
                      segment_boundaries=est["taus"])

    fig.suptitle(f"{cfg.name} — Varying PINN", fontsize=13)
    _save(fig, output_path)


### Plotting for steady state algorithm
def reconstruct_CO2_steady_state(t, Q, S, taus, C0, V, C_out):
    """
    Reconstruct CO2 trajectory from steady-state pipeline Q/S estimates.
    Supports piecewise segments.
    """

    t = np.asarray(t)
    Q = np.asarray(Q)
    S = np.asarray(S)
    taus = np.asarray(taus)

    boundaries = np.concatenate(([t[0]], taus, [t[-1]]))

    C_pred = np.zeros_like(t, dtype=float)
    C_current = C0

    for i in range(len(Q)):
        mask = (t >= boundaries[i]) & (t <= boundaries[i + 1])

        if not np.any(mask):
            continue

        t_seg = t[mask]

        tau = V / Q[i]
        C_ss = C_out + S[i] / Q[i]

        C_seg = C_ss + (C_current - C_ss) * np.exp(
            -(t_seg - t_seg[0]) / tau
        )

        C_pred[mask] = C_seg
        C_current = C_seg[-1]

    return C_pred

def plot_steady_state_CO2(ax, t, C_meas, result, cfg):
    """
    Plot measured CO2 against reconstructed CO2.
    """

    ax.scatter(t, C_meas, s=8, alpha=0.4, label="Measured CO2")

    C_pred = reconstruct_CO2_steady_state(
        t,
        result["Q"],
        result["S"],
        result.get("taus", []),
        C_meas[0],
        cfg.physics.V,
        cfg.physics.C_out,
    )

    ax.plot(t, C_pred, linewidth=2, label="Estimated CO2")

    style_ax(ax, "CO2 Reconstruction", "Time [h]", "CO2")

def plot_steady_state_classification(ax, t, C, steady_mask):
    ax.plot(t, C, lw=1.5, label="CO2")
    ax.scatter(t[steady_mask], C[steady_mask], s=8, label="steady")
    ax.scatter(t[~steady_mask], C[~steady_mask], s=8, label="transient")
    style_ax(ax, "Steady/Transient Classification", "Time [h]", "CO2")


def plot_steady_state_parameters(ax_Q, ax_S, result, truth=None):
    n = len(result["Q"])
    x = np.arange(n)

    ax_Q.plot(x, result["Q"], marker="o", label="Estimated Q")
    if truth is not None:
        ax_Q.plot(x, truth["Q"], marker="x", ls="--", label="True Q")
    style_ax(ax_Q, "Q Estimates", "Segment", "Q")

    ax_S.plot(x, result["S"], marker="o", label="Estimated S")
    if truth is not None:
        ax_S.plot(x, truth["S"], marker="x", ls="--", label="True S")
    style_ax(ax_S, "S Estimates", "Segment", "S")


def plot_steady_state_results(result, data, cfg, steady_mask, truth=None, output_path="steady_state.png"):
    t = data["t_np"].flatten()
    C = data["c_np"].flatten()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    plot_steady_state_classification(axes[0], t, C, steady_mask)
    plot_steady_state_parameters(axes[1], axes[2], result, truth)

    fig.suptitle(f"{cfg.name} — Steady State Pipeline", fontsize=13)
    _save(fig, output_path)