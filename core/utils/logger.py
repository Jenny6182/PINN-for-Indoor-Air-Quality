"""
logger.py
---------
History dict factories, log_fn callbacks, and terminal printing.

Each log_fn has signature: log_fn(model, history, epoch)
and is passed into train_loop as the log_fn argument.

Usage:
    from logger import make_history_simple, log_fn_simple, print_header, print_row

    history = make_history_simple()
    train_loop(..., history=history, log_fn=log_fn_simple, ...)
"""

import torch
import numpy as np


# ── history factories ─────────────────────────────────────────────────────────

def make_history_simple():
    """For constant PINN — tracks scalar Q and S per epoch."""
    return {
        "loss_total": [],
        "loss_data":  [],
        "loss_phys":  [],
        "Q":          [],
        "S":          [],
    }


def make_history_varying():
    """For piecewise PINN — tracks mean Q/S and full segment arrays per epoch."""
    return {
        "loss_total":  [],
        "loss_data":   [],
        "loss_phys":   [],
        "Q_mean":      [],
        "S_mean":      [],
        "Q_segments":  [],
        "S_segments":  [],
    }


def make_history_stage2():
    """For RAA-PINN Stage II — tracks tau and Q/S on each side per epoch."""
    return {
        "loss_total": [],
        "loss_data":  [],
        "loss_phys":  [],
        "tau":        [],
        "Q_minus":    [],
        "Q_plus":     [],
        "S_minus":    [],
        "S_plus":     [],
    }


def make_history_one_pinn():
    """For one single RAA-PINN covering entire domain — tracks all taus and Q/S per epoch."""
    return {
        "loss_total": [],
        "loss_data":  [],
        "loss_phys":  [],
        "taus":       [],      # list of arrays, one per epoch
        "Q_minus":    [],      # list of arrays (before each changepoint)
        "Q_plus":     [],      # list of arrays (after each changepoint)
        "S_minus":    [],      # list of arrays (before each changepoint)
        "S_plus":     [],      # list of arrays (after each changepoint)
    }


# ── log_fn callbacks ──────────────────────────────────────────────────────────

def log_fn_simple(model, history, epoch):
    """
    Appends scalar Q and S to history each epoch.
    Expects model.param_model to be ConstantParams.
    Called inside train_loop after standard losses are appended.
    """
    with torch.no_grad():
        # pass a dummy time — ConstantParams ignores t_phys anyway
        t_dummy  = torch.tensor([[0.0]])
        Q, S     = model.param_model.get_Q_S(t_dummy)
        history["Q"].append(Q.item())
        history["S"].append(S.item())


def log_fn_varying(model, history, epoch):
    """
    Appends mean Q/S and full segment arrays to history each epoch.
    Expects model.param_model to be SegmentParams.
    """
    with torch.no_grad():
        Q_seg = torch.exp(model.param_model.log_Q_segments).detach().numpy().copy()
        S_seg = torch.exp(model.param_model.log_S_segments).detach().numpy().copy()
        history["Q_mean"].append(float(Q_seg.mean()))
        history["S_mean"].append(float(S_seg.mean()))
        history["Q_segments"].append(Q_seg)
        history["S_segments"].append(S_seg)


def log_fn_stage2(model, history, epoch):
    """
    Appends tau and Q/S on each side of the jump to history each epoch.
    Expects model.param_model to be SigmoidChangepoint.
    """
    with torch.no_grad():
        pm = model.param_model
        history["tau"].append(pm.tau.item())
        history["Q_minus"].append(torch.exp(pm.log_Q_minus).item())
        history["Q_plus"].append(torch.exp(pm.log_Q_plus).item())
        history["S_minus"].append(torch.exp(pm.log_S_minus).item())
        history["S_plus"].append(torch.exp(pm.log_S_plus).item())

def log_fn_one_pinn(model, history, epoch):
    """
    Appends all taus and Q/S values (before and after each changepoint) to history each epoch.
    Expects model.param_model to be MultiSigmoidChangepoint.
    For K changepoints:
      - taus: array of K changepoint times
      - Q_minus/Q_plus: arrays of K values (Q before and after each changepoint)
      - S_minus/S_plus: arrays of K values (S before and after each changepoint)
    """
    with torch.no_grad():
        pm = model.param_model
        # Get all taus (K changepoints)
        taus = pm.taus.detach().cpu().numpy()
        # Get all Q and S segment values (K+1 segments)
        Q_seg = torch.exp(torch.stack(list(pm.log_Q))).detach().cpu().numpy()
        S_seg = torch.exp(torch.stack(list(pm.log_S))).detach().cpu().numpy()
        
        # For each changepoint i: Q_minus=Q[i], Q_plus=Q[i+1]
        Q_minus = Q_seg[:-1]  # segments 0 to K-1
        Q_plus = Q_seg[1:]    # segments 1 to K
        S_minus = S_seg[:-1]
        S_plus = S_seg[1:]
        
        history["taus"].append(taus)
        history["Q_minus"].append(Q_minus)
        history["Q_plus"].append(Q_plus)
        history["S_minus"].append(S_minus)
        history["S_plus"].append(S_plus)


# ── terminal printing ─────────────────────────────────────────────────────────

def print_header(pinn_type="simple"):
    """Print the column header before the training loop starts."""
    if pinn_type == "simple":
        print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  "
              f"{'Q':>10}  {'S':>12}")
    elif pinn_type == "varying":
        print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  "
              f"{'Q_mean':>10}  {'S_mean':>12}")
    elif pinn_type == "stage2":
        print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  "
              f"{'tau':>8}  {'Q-':>8}  {'Q+':>8}  {'S-':>10}  {'S+':>10}")
    elif pinn_type == "one_pinn":
        print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}  "
              f"{'tau(1st)':>15}  {'Q-':>8}  {'Q+':>8}  {'S-':>10}  {'S+':>10}")
    print("-" * 75)


def print_row(epoch, history, pinn_type="simple"):
    """
    Print one progress row using the latest values in history.
    Call this inside the training loop at whatever print_every interval you want.
    """
    loss      = history["loss_total"][-1]
    loss_data = history["loss_data"][-1]
    loss_phys = history["loss_phys"][-1]

    base = (f"{epoch:>6}  {loss:>10.4e}  "
            f"{loss_data:>10.4e}  {loss_phys:>10.4e}")

    if pinn_type == "simple":
        print(f"{base}  {history['Q'][-1]:>10.2f}  {history['S'][-1]:>12.2e}")

    elif pinn_type == "varying":
        print(f"{base}  {history['Q_mean'][-1]:>10.2f}  {history['S_mean'][-1]:>12.2e}")

    elif pinn_type == "stage2":
        print(f"{base}  {history['tau'][-1]:>8.3f}  "
              f"{history['Q_minus'][-1]:>8.1f}  {history['Q_plus'][-1]:>8.1f}  "
              f"{history['S_minus'][-1]:>10.2e}  {history['S_plus'][-1]:>10.2e}")

    elif pinn_type == "one_pinn":
        # For one_pinn with multiple changepoints, display first changepoint and indicate count
        taus = history["taus"][-1]  # array of K changepoints
        Q_minus = history["Q_minus"][-1]
        Q_plus = history["Q_plus"][-1]
        S_minus = history["S_minus"][-1]
        S_plus = history["S_plus"][-1]
        n_changepoints = len(taus)
        # Display first changepoint
        print(f"{base}  {taus[0]:>8.3f} ({n_changepoints}cp)  "
              f"{Q_minus[0]:>8.1f}  {Q_plus[0]:>8.1f}  "
              f"{S_minus[0]:>10.2e}  {S_plus[0]:>10.2e}")