"""
Stage 2: closed-form parameter estimation per segment.
Each segment is solved independently using the analytical solution 
for a single well-mixed zone with piecewise-constant Q and S:

    V dC/dt = Q(C_out - C) + S
    C(t) = Css + (C0 - Css) * exp(-t/tau),  Css = C_out + S/Q,  tau = V/Q

Per segment:
  Css  = mean of the last portion of the steady-labeled points (where it possibly plateaus),
         not the whole steady region; see estimate_css()
  tau  = fit via nonlinear least squares (curve_fit) on
         C(t) = Css_fit + (C0 - Css_fit)*exp(-t/tau), with C0 fixed to
         the known first transient sample and Css_fit seeded from (but
         not locked to) the plateau estimate; see estimate_tau().
  Q    = V / tau
  S    = Q * (Css - C_out)
"""


from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


@dataclass
class SegmentEstimate:
    """Container for one segment's estimated parameters + fit diagnostics."""
    start_idx: int
    end_idx: int
    n_steady: int     # number of points labeled steady in this segment
    n_transient: int  # number of points used in the tau fit
    Css: float        # estimated steady-state concentration
    tau: float        # estimated time constant (hours)
    Q: float          # estimated ventilation rate, Q = V/tau
    S: float          # estimated source rate, S = Q*(Css - C_out)
    tau_fit_r2: float # goodness of fit for the tau regression


def estimate_css(t, C, steady_idx, fraction=0.2):
    """
    Estimate the steady-state concentration Css for a segment.
    Only the last "fraction (the fraction parameter)" of the 
    steady-labeled points are used to determine the plateau instead of
    the whole steady region. 
    Early points in a steady region can still be mid-convergence if 
    the segmentation's slope threshold labeled them "steady" too early
    Averaging those in would bias Css away from the true asymptote.
    """
    if len(steady_idx) == 0:
        return np.nan

    n = max(1, int(len(steady_idx) * fraction))
    plateau_idx = steady_idx[-n:]

    return float(np.mean(C[plateau_idx]))


def estimate_tau(t, C, transient_idx, Css):
    """
    Estimate tau (and refine Css) for one segment via nonlinear least
    squares on C(t) directly (not the log-linearized form).

    Design notes:
      - C0 is fixed to the first sample of the transient window (it's
        known exactly from the data).
      - Css is seeded from the plateau estimate as p0 and then fitted
        with tau jointly
      - Time is shifted to start at 0 (tt - tt[0]) so the fit is always
        working in local segment time; this only affects numerical
        conditioning, not the fitted tau itself.
      - "transient_idx" is expected to be the full window between the
        old and new steady regions after changepoint refinement.

    Returns (tau, r2); (nan, nan) if the segment can't be fit.
    """
    if len(transient_idx) < 2:
        return np.nan, np.nan

    tt = np.asarray(t)[transient_idx]
    tt = tt - tt[0]  # local time, starting at 0
    CC = np.asarray(C)[transient_idx]
    C0 = CC[0]  # known exactly, not fitted

    # plotting the data fed into estimate_tau: confirm what window was handed to the fit
    plt.figure(figsize=(6, 4))
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

    p0 = [Css, 0.25]  # seed Css_fit with the plateau estimate, tau with a rough guess

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


def estimate_segment(t, C, seg, V, C_out):
    """
    Estimate Css, tau, Q, S (+ fit diagnostics) for a single segment,
    and package the result as a SegmentEstimate.
    """
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