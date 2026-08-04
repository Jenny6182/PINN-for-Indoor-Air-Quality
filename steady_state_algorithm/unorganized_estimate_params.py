"""
Stage 2 replacement: closed-form / linear-regression parameter
estimation per segment. No PINN, no nonlinear solver.

  Css       = mean of the steady-labeled points
  tau       = -1 / slope of linear regression of ln|C(t) - Css| vs t,
              using only the transient-labeled points
  Q (=L)    = V / tau
  S         = Q * (Css - C_out)
"""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SegmentEstimate:
    start_idx: int
    end_idx: int
    n_steady: int
    n_transient: int
    Css: float
    tau: float
    Q: float
    S: float
    tau_fit_r2: float

def estimate_css(t, C, steady_idx, fraction=0.2):
    if len(steady_idx) == 0:
        return np.nan

    # use only the final portion of the steady region
    n = max(1, int(len(steady_idx) * fraction))
    plateau_idx = steady_idx[-n:]

    return float(np.mean(C[plateau_idx]))

from scipy.optimize import curve_fit

def estimate_tau(t, C, transient_idx, Css): # large tau version, problematic
    if len(transient_idx) < 2:
        return np.nan, np.nan

    tt = np.asarray(t)[transient_idx]
    tt = tt - tt[0]
    CC = np.asarray(C)[transient_idx]
    C0 = CC[0]  # known exactly, not fitted

    plt.figure(figsize=(6,4))
    plt.plot(tt, CC, marker="o")
    plt.axhline(Css, linestyle="--", label="Css")
    plt.xlabel("time")
    plt.ylabel("CO2")
    plt.title("Data passed into estimate_tau")
    plt.legend()
    plt.grid()
    plt.show()

    def model(t, Css_fit, tau):
        return Css_fit + (C0 - Css_fit) * np.exp(-t / tau)

    p0 = [Css, 0.25]

    try:
        popt, pcov = curve_fit(model, tt, CC, p0=p0, maxfev=5000)
    except RuntimeError:
        return np.nan, np.nan

    Css_fit, tau = popt
    if tau <= 0:
        return np.nan, np.nan

    residuals = CC - model(tt, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((CC - np.mean(CC))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print("t range:", tt[0], tt[-1])
    print("C range:", CC[0], CC[-1])
    print("Css guess:", Css)

    return float(tau), float(r2)

def get_full_transient_idx(seg):
    """
    Transient = everything between start of change and final steady state.
    """
    transient_start = seg.steady_idx[-1] + 1  # after old steady region
    transient_end = seg.steady_idx_next[0] - 1  # before new steady region

    return np.arange(transient_start, transient_end + 1)

def estimate_segment(t, C, seg, V, C_out):
    Css = estimate_css(t, C, seg.steady_idx)
    tau, r2 = estimate_tau(t, C, seg.transient_idx, Css)

    Q = V / tau if tau and not np.isnan(tau) else np.nan
    S = Q * (Css - C_out) if Q and not np.isnan(Q) and not np.isnan(Css) else np.nan

    return SegmentEstimate(
        start_idx=seg.start_idx,
        end_idx=seg.end_idx,
        n_steady=len(seg.steady_idx),
        n_transient=len(seg.transient_idx),
        Css=Css,
        tau=tau,
        Q=Q,
        S=S,
        tau_fit_r2=r2,
    )




