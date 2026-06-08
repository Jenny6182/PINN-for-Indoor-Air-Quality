"""
Simple IAQ Scenario — Constant Source, Constant Ventilation
===========================================================

Single well-mixed zone:

    V dC/dt = Q (C_out - C) + S

Analytical solution:

    C(t) = C_ss + (C0 - C_ss) exp(-Qt/V)
    C_ss = C_out + S/Q
    tau  = V/Q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def run_iaq_simulation(
    V=100.0,
    Q=500.0,
    S_vol=0.10,
    C_out=420.0,
    C0=500.0,
    duration_h=8.0,
    dt_min=1.0,
    sigma_meas=15.0,
    seed=0,
    save_csv=True,
    save_plot=True,
    out_dir="datasets",
):
    """
    Run constant-Q constant-S IAQ simulation.

    Parameters
    ----------
    V : float - Zone volume [m^3]

    Q : float - Ventilation airflow [m^3/h]

    S_vol : float - CO2 generation rate [m^3 CO2 / h]

    Returns
    -------
    dict containing:
        df          : DataFrame
        C_true      : ODE solution
        C_analytic  : analytical solution
        C_meas      : noisy measurements
        C_ss        : steady-state concentration
        tau         : time constant
    """

    # Convert source term
    S = 1e6 * S_vol

    # Time grid
    n_steps = int(duration_h * 60 / dt_min) + 1
    t_hours = np.linspace(0.0, duration_h, n_steps)

    # ODE
    def rhs(t, C):
        return [(Q * (C_out - C[0]) + S) / V]

    sol = solve_ivp(rhs,(t_hours[0], t_hours[-1]), [C0], t_eval=t_hours, method="RK45", rtol=1e-8, atol=1e-4,)
    C_true = sol.y[0]

    # Analytical solution
    C_ss = C_out + S / Q
    tau = V / Q

    C_analytic = (C_ss + (C0 - C_ss) * np.exp(-Q * t_hours / V))

    # Add noise
    rng = np.random.default_rng(seed)
    C_meas = C_true + rng.normal(0.0, sigma_meas, size=C_true.shape,)

    # Dataframe
    df = pd.DataFrame({"t_hours": t_hours, "C_true_ppm": C_true, "C_analytic_ppm": C_analytic, "C_meas_ppm": C_meas,})

    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    Q_int = int(Q)

    # Save CSV
    if save_csv:
        csv_name = f"iaq_Q{Q_int}_S{S_vol:.2f}.csv"
        df.to_csv(OUT / csv_name, index=False)

    # Plot
    if save_plot:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(t_hours, C_meas, s=8, alpha=0.4, color="#888", label="Measured")
        ax.plot(t_hours, C_true, color="#c0392b", lw=2, label="ODE")
        ax.plot(t_hours, C_analytic, "--", color="black", lw=1.2, label="Analytical")

        ax.axhline(C_ss, color="#2c7fb8", ls=":", label=f"C_ss={C_ss:.0f}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("CO₂ [ppm]")

        ax.set_title(f"Q={Q:.0f}, S_vol={S_vol:.2f}, τ={tau*60:.1f} min")

        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        png_name = f"iaq_Q{Q_int}_S{S_vol:.2f}.png"
        fig.savefig(OUT / png_name, dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"Q={Q}, S_vol={S_vol}, C_ss={C_ss:.1f}, tau={tau:.3f} h")

        return {
        "df": df,
        "C_true": C_true,
        "C_analytic": C_analytic,
        "C_meas": C_meas,
        "C_ss": C_ss,
        "tau": tau,
        }


# Optional default execution
if __name__ == "__main__":
    run_iaq_simulation()