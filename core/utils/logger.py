"""
logger.py
---------
History tracking and terminal printing for PINN training.

Provides a single history factory and log function that work across all
PINN variants, and print utilities for monitoring training progress.
"""

import torch
import numpy as np

def make_history() -> dict:
    return {
        "loss_total": [],
        "loss_data":  [],
        "loss_phys":  [],
        "Q":          [], # array per epoch
        "S":          [], # array per epoch
        "taus":       [] # none or array per epoch
    }

def log_fn(model, history, loss, loss_data, loss_phys):
    """
    Train loop calls this function at certain number of epochs to 
    record current estimates of Q, S, and tau,
    and current physics loss, data loss, total loss
    """
    with torch.no_grad():
        estimates = model.param_model.get_final_estimates()
        
        history["Q"].append(estimates["Q"])
        history["S"].append(estimates["S"])
        history["taus"].append(estimates["taus"])

        history["loss_total"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_phys"].append(loss_phys.item())


def print_header():
    """
    Prints header for each run
    """
    print(f"{'Epoch':>6}  {'Loss':>10}  {'Data':>10}  {'Phys':>10}")
    print(f"{'------':>6}  {'----------':>10}  {'----------':>10}  {'----------':>10}")


def print_row(epoch, history):
    """
    Prints each row during a run
    """
    Q    = history["Q"][-1]
    S    = history["S"][-1]
    taus = history["taus"][-1]

    print(f"{epoch:>6}  {history['loss_total'][-1]:>10.4e}  "
          f"{history['loss_data'][-1]:>10.4e}  {history['loss_phys'][-1]:>10.4e}")
    print(f"  Q    = {Q}")
    print(f"  S    = {S}")
    if taus is not None:
        print(f"  taus = {taus}")
    print()