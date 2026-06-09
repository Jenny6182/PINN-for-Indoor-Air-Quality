"""
PINN to Estimate Piecewise-Constant IAQ CO2 Parameters
-------------------------------------------------------
Extension of the constant Q/S PINN to handle piecewise-constant Q(t) and S(t).

Instead of two scalar parameters log_Q, log_S, we now have two vectors:
    log_Q_segments: shape (n_segments,)  — one log_Q per 0.5h segment
    log_S_segments: shape (n_segments,)  — one log_S per 0.5h segment

Everything else — warm-up, ramp, auto_scale, normalisation — stays the same.

The dataset CSV must have columns:
    t_hours, C_true_ppm, C_meas_ppm, Q_true, S_true

which is exactly what iaq_co2_piecewise.py produces.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

# ------ data loading ------
def load_data(path):
    df = pd.read_csv(path)

    t_np      = df["t_hours"].values.reshape(-1, 1).astype(np.float32)
    c_np      = df["C_meas_ppm"].values.reshape(-1, 1).astype(np.float32)
    Q_true_np = df["Q_true"].values.astype(np.float32)   # ground truth per timestep
    S_true_np = df["S_true"].values.astype(np.float32)

    t_train_np, t_test_np, c_train_np, c_test_np = train_test_split(
        t_np, c_np, test_size=0.2, random_state=42
    )
    return t_np, c_np, t_train_np, c_train_np, t_test_np, c_test_np, Q_true_np, S_true_np


# ------ normalisation helpers ------
def norm_t(t, t_min, t_max):
    return (t - t_min) / (t_max - t_min + 1e-8)

def norm_c(c, c_mean, c_std):
    return (c - c_mean) / c_std

def denorm_c(c_hat, c_mean, c_std):
    return c_hat * c_std + c_mean

def find_stats(t_train_np, c_train_np):
    return (t_train_np.min(), t_train_np.max(),
            c_train_np.mean(), c_train_np.std())

def normalize_data(t_train_np, c_train_np, t_min, t_max, c_mean, c_std):
    T = torch.tensor(norm_t(t_train_np, t_min, t_max), dtype=torch.float32)
    C = torch.tensor(norm_c(c_train_np, c_mean, c_std), dtype=torch.float32)
    return T, C

def create_collocation_points(t_min, t_max, duration_h, n_segments):
    """
    Uniform grid plus extra points just around each segment boundary,
    so the physics loss captures the behaviour at each Q/S jump.
    """
    t_uniform = np.linspace(t_min, t_max, N_COLLOC)

    boundaries = np.arange(1, n_segments) * SEGMENT_DURATION
    # only keep boundaries within [t_min, t_max]
    boundaries = boundaries[(boundaries > t_min) & (boundaries < t_max)]
    t_near = np.concatenate([boundaries - 0.005, boundaries + 0.005])

    t_col_np = np.sort(np.unique(np.concatenate([t_uniform, t_near])))
    t_col_np = np.clip(t_col_np, t_min, t_max).reshape(-1, 1).astype(np.float32)

    T_col = torch.tensor(
        norm_t(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True
    )
    return t_col_np, T_col


#------PINN model ------
class PINN(nn.Module):
    def __init__(self, n_segments, hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN):
        super().__init__()

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

        # one log_Q and one log_S per segment — vectors now, not scalars
        self.log_Q_segments = nn.Parameter(
            torch.full((n_segments,), LOG_Q_INIT, dtype=torch.float32)
        )
        self.log_S_segments = nn.Parameter(
            torch.full((n_segments,), LOG_S_INIT, dtype=torch.float32)
        )

    def forward(self, t_norm):
        return self.net(t_norm)

    @property
    def Q_seg(self):
        return torch.exp(self.log_Q_segments)   # shape (n_segments,)

    @property
    def S_seg(self):
        return torch.exp(self.log_S_segments)   # shape (n_segments,)


# ------ physics residual ------
def physics_residual(model, T_col, t_min, t_max, c_mean, c_std, n_segments):
    """
    Same normalised PDE as the constant case, but Q and S are now looked up
    per collocation point based on which segment it falls in.
    """
    C_norm_pred = model(T_col)

    dC_dt_norm = torch.autograd.grad(
        C_norm_pred, T_col,
        grad_outputs=torch.ones_like(C_norm_pred),
        create_graph=True,
    )[0]

    # convert normalised time → physical hours → segment index
    t_phys  = T_col.detach() * (t_max - t_min) + t_min          # (N, 1)
    # seg_idx = torch.clamp(
    #     (t_phys / SEGMENT_DURATION).long(), 0, n_segments - 1
    # ).squeeze(1)                                                  # (N,)

    # look up Q and S for each collocation point
    # Q_col = model.Q_seg[seg_idx].unsqueeze(1)   # (N, 1)
    # S_col = model.S_seg[seg_idx].unsqueeze(1)   # (N, 1)

    dt         = float(t_max - t_min)
    alpha      = (Q_col / V) * dt
    beta       = (S_col / (V * c_std)) * dt
    C_out_norm = (C_out - c_mean) / c_std

    rhs      = alpha * (C_out_norm - C_norm_pred) + beta
    residual = dC_dt_norm - rhs
    return residual


# ------ training loop ------
def train_loop(model, opt_net, opt_params, sched_net, sched_params,
               T_train, C_train, T_col,
               t_min, t_max, c_mean, c_std, n_segments, history):

    phys_loss_init     = None
    auto_scale_frozen  = None

    print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  "
          f"{'Q_mean':>8}  {'S_mean':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):

        opt_net.zero_grad()
        opt_params.zero_grad()

        # data loss
        C_pred_train = model(T_train)
        loss_data    = torch.mean((C_pred_train - C_train) ** 2)

        # physics loss
        if epoch <= WARMUP_EPOCHS:
            loss_phys = torch.tensor(0.0)
            lam       = 0.0
        else:
            residual  = physics_residual(
                model, T_col, t_min, t_max, c_mean, c_std, n_segments
            )
            loss_phys = torch.mean(residual ** 2)

            # capture scale once on the very first physics epoch, then freeze
            if phys_loss_init is None:
                phys_loss_init    = loss_phys.detach().item()
                auto_scale_frozen = loss_data.detach().item() / (phys_loss_init + 1e-8)

            ramp_frac = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
            lam       = LAMBDA_PHYS * ramp_frac * auto_scale_frozen

        loss = loss_data + lam * loss_phys

        loss.backward()
        opt_net.step()
        opt_params.step()
        sched_net.step()
        sched_params.step()

        with torch.no_grad():
            q_mean = model.Q_seg.mean().item()
            s_mean = model.S_seg.mean().item()

        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())
        history["Q_mean"].append(q_mean)
        history["S_mean"].append(s_mean)
        history["Q_segments"].append(model.Q_seg.detach().numpy().copy())
        history["S_segments"].append(model.S_seg.detach().numpy().copy())

        if epoch % 500 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>10.4e}  {loss_data.item():>10.4e}  "
                  f"{loss_phys.item():>10.4e}  {q_mean:>8.1f}  {s_mean:>10.1f}")

    return history


# ------diagnostics plot ─------
def plot_diagnostics(model, history, t_np, c_np, t_train_np, c_train_np,
                     t_col_np, Q_true_np, S_true_np,
                     t_min, t_max, c_mean, c_std, n_segments, output_path):

    epochs_arr = np.arange(1, EPOCHS + 1)

    fig = plt.figure(figsize=(15, 18))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.35)

    # --- row 0: loss curves -----------
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.semilogy(epochs_arr, history["loss_total"], lw=1.5, color="#2c3e50", label="Total")
    ax_loss.semilogy(epochs_arr, history["loss_data"],  lw=1.5, color="#e74c3c", ls="--", label="Data")
    ax_loss.semilogy(epochs_arr, history["loss_phys"],  lw=1.5, color="#3498db", ls="--", label="Physics")
    ax_loss.axvline(WARMUP_EPOCHS, color="gray", ls=":", lw=1, label=f"Warm-up end ({WARMUP_EPOCHS})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.set_title("Training Losses")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    # --- row 1: mean Q and S convergence ------
    ax_Q = fig.add_subplot(gs[1, 0])
    ax_Q.plot(epochs_arr, history["Q_mean"], lw=1.5, color="#e67e22", label="Q mean (estimated)")
    ax_Q.axhline(Q_true_np.mean(), color="black", ls="--", lw=1,
                 label=f"Q true mean = {Q_true_np.mean():.1f}")
    ax_Q.set_xlabel("Epoch")
    ax_Q.set_ylabel("Q [m³/h]")
    ax_Q.set_title("Mean Q over Training")
    ax_Q.legend(fontsize=9)
    ax_Q.grid(alpha=0.3)

    ax_S = fig.add_subplot(gs[1, 1])
    ax_S.plot(epochs_arr, history["S_mean"], lw=1.5, color="#8e44ad", label="S mean (estimated)")
    ax_S.axhline(S_true_np.mean(), color="black", ls="--", lw=1,
                 label=f"S true mean = {S_true_np.mean():.2e}")
    ax_S.set_xlabel("Epoch")
    ax_S.set_ylabel("S [ppm·m³/h]")
    ax_S.set_title("Mean S over Training")
    ax_S.legend(fontsize=9)
    ax_S.grid(alpha=0.3)

    # --- row 2: per-segment Q and S at end of training ----
    seg_ids   = np.arange(n_segments)
    t_centers = (seg_ids + 0.5) * SEGMENT_DURATION

    # get unique true values per segment from the arrays
    Q_true_seg = np.array([
        Q_true_np[np.searchsorted(
            np.linspace(t_np.min(), t_np.max(), len(Q_true_np)),
            t_centers[i], side="left"
        )] for i in range(n_segments)
    ])
    # cleaner: just take the true value at each segment centre from the df columns
    # using the stored Q_true_np and S_true_np arrays (sampled at t_hours)
    # map segment index -> representative true value
    t_hours_all = np.linspace(t_np.min(), t_np.max(), len(Q_true_np))
    Q_true_seg  = np.array([
        Q_true_np[np.argmin(np.abs(t_hours_all - tc))] for tc in t_centers
    ])
    S_true_seg  = np.array([
        S_true_np[np.argmin(np.abs(t_hours_all - tc))] for tc in t_centers
    ])

    Q_est_seg = model.Q_seg.detach().numpy()
    S_est_seg = model.S_seg.detach().numpy()

    ax_Qseg = fig.add_subplot(gs[2, 0])
    ax_Qseg.bar(seg_ids - 0.2, Q_true_seg, width=0.4, color="#2980b9", alpha=0.7, label="Q true")
    ax_Qseg.bar(seg_ids + 0.2, Q_est_seg,  width=0.4, color="#e67e22", alpha=0.7, label="Q estimated")
    ax_Qseg.set_xlabel("Segment index")
    ax_Qseg.set_ylabel("Q [m³/h]")
    ax_Qseg.set_title("Per-Segment Q: True vs Estimated")
    ax_Qseg.legend(fontsize=9)
    ax_Qseg.grid(alpha=0.3, axis="y")

    ax_Sseg = fig.add_subplot(gs[2, 1])
    ax_Sseg.bar(seg_ids - 0.2, S_true_seg / 1e6, width=0.4, color="#27ae60", alpha=0.7, label="S_vol true")
    ax_Sseg.bar(seg_ids + 0.2, S_est_seg  / 1e6, width=0.4, color="#8e44ad", alpha=0.7, label="S_vol estimated")
    ax_Sseg.set_xlabel("Segment index")
    ax_Sseg.set_ylabel("S_vol [m³ CO₂/h]")
    ax_Sseg.set_title("Per-Segment S_vol: True vs Estimated")
    ax_Sseg.legend(fontsize=9)
    ax_Sseg.grid(alpha=0.3, axis="y")

    # --- row 3: predicted C(t) vs measurements --------
    ax_fit = fig.add_subplot(gs[3, :])

    t_full_np = np.linspace(t_np.min(), t_np.max(), 500).reshape(-1, 1).astype(np.float32)
    T_full    = torch.tensor(norm_t(t_full_np, t_min, t_max), dtype=torch.float32)
    with torch.no_grad():
        C_pred_norm = model(T_full).numpy()
    C_pred_phys = C_pred_norm * c_std + c_mean

    ax_fit.scatter(t_np, c_np, s=6, alpha=0.3, color="#aaa", label="C_meas (all)")
    ax_fit.scatter(t_train_np, c_train_np, s=8, alpha=0.5, color="#e74c3c", label="C_meas (train)")
    ax_fit.plot(t_full_np, C_pred_phys, lw=2, color="#2c3e50", label="PINN prediction")
    ax_fit.set_xlabel("Time [h]")
    ax_fit.set_ylabel("CO₂ [ppm]")
    ax_fit.set_title("PINN Prediction vs Measurements")
    ax_fit.legend(fontsize=9)
    ax_fit.grid(alpha=0.3)

    # --- row 4: PDE residual -----------
    ax_res = fig.add_subplot(gs[4, :])

    T_col_eval = torch.tensor(
        norm_t(t_col_np, t_min, t_max), dtype=torch.float32, requires_grad=True
    )
    res_eval = physics_residual(
        model, T_col_eval, t_min, t_max, c_mean, c_std, n_segments
    )
    res_np = res_eval.detach().numpy().flatten()

    ax_res.plot(t_col_np.flatten(), res_np, lw=1, color="#c0392b", alpha=0.8)
    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.fill_between(t_col_np.flatten(), res_np, alpha=0.15, color="#c0392b")
    # mark segment boundaries
    for b in np.arange(1, n_segments) * SEGMENT_DURATION:
        ax_res.axvline(b, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax_res.set_xlabel("Time [h]")
    ax_res.set_ylabel("PDE Residual (normalised)")
    ax_res.set_title("Physics Residual")
    ax_res.grid(alpha=0.3)

    fig.suptitle("Piecewise PINN Diagnostics — IAQ CO₂ Parameter Recovery",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"Saved figure → {output_path}")
    plt.show()


# ------ main ------
def main(path, plot_output_path=None):
    print(f"\nFile: {path}")

    (t_np, c_np,
     t_train_np, c_train_np,
     t_test_np, c_test_np,
     Q_true_np, S_true_np) = load_data(path)

    t_min, t_max, c_mean, c_std = find_stats(t_train_np, c_train_np)

    T_train, C_train = normalize_data(t_train_np, c_train_np, t_min, t_max, c_mean, c_std)

    duration_h  = float(t_np.max() - t_np.min())
    n_segments  = int(round(duration_h / SEGMENT_DURATION))
    print(f"  duration={duration_h:.1f}h, segment_duration={SEGMENT_DURATION}h, "
          f"n_segments={n_segments}")

    t_col_np, T_col = create_collocation_points(t_min, t_max, duration_h, n_segments)

    model = PINN(n_segments=n_segments)

    net_params   = list(model.net.parameters())
    phys_params  = [model.log_Q_segments, model.log_S_segments]

    # separate optimisers so schedulers can be tuned independently
    opt_net    = torch.optim.Adam(net_params,  lr=LR_NET)
    opt_params = torch.optim.Adam(phys_params, lr=LR_PARAMS)

    sched_net    = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_net,    T_max=EPOCHS, eta_min=1e-5   # network decays low
    )
    sched_params = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_params, T_max=EPOCHS, eta_min=1e-3   # params stay mobile
    )

    history = {
        "loss_total": [], "loss_data": [], "loss_phys": [],
        "Q_mean": [], "S_mean": [],
        "Q_segments": [],   # list of arrays, one per epoch
        "S_segments": [],
    }

    history = train_loop(
        model, opt_net, opt_params, sched_net, sched_params,
        T_train, C_train, T_col,
        t_min, t_max, c_mean, c_std, n_segments, history,
    )

    print("\nDone.")
    print(f"  True Q range:      {Q_true_np.min():.1f} – {Q_true_np.max():.1f}")
    print(f"  Estimated Q range: {model.Q_seg.min().item():.1f} – {model.Q_seg.max().item():.1f}")
    print(f"  True S range:      {S_true_np.min():.2e} – {S_true_np.max():.2e}")
    print(f"  Estimated S range: {model.S_seg.min().item():.2e} – {model.S_seg.max().item():.2e}")

    print("\n  Per-segment results:")
    print(f"  {'seg':>4}  {'t_start':>8}  {'Q_true':>8}  {'Q_est':>8}  {'S_true':>10}  {'S_est':>10}")

    Q_est_seg = model.Q_seg.detach().numpy()
    S_est_seg = model.S_seg.detach().numpy()

    # get true value per segment from the arrays
    t_hours_all = np.linspace(t_np.min(), t_np.max(), len(Q_true_np))
    for i in range(n_segments):
        t_center = (i + 0.5) * SEGMENT_DURATION
        idx      = np.argmin(np.abs(t_hours_all - t_center))
        print(f"  {i:>4}  {i*SEGMENT_DURATION:>8.2f}  "
            f"{Q_true_np[idx]:>8.1f}  {Q_est_seg[i]:>8.1f}  "
            f"{S_true_np[idx]:>10.2e}  {S_est_seg[i]:>10.2e}")

    stem        = Path(path).stem
    output_path = plot_output_path or f"iaq_pinn_piecewise_{stem}.png"

    plot_diagnostics(
        model, history,
        t_np, c_np, t_train_np, c_train_np,
        t_col_np, Q_true_np, S_true_np,
        t_min, t_max, c_mean, c_std, n_segments,
        output_path,
    )


if __name__ == "__main__":
    # run on both piecewise datasets
    main("varying_datasets/iaq_co2_varying_Q.csv")
    main("varying_datasets/iaq_co2_varying_S.csv")