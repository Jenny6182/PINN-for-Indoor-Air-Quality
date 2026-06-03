"""
Simple IAQ Scenario — Constant Source, Constant Ventilation
============================================================

Single well-mixed zone with all parameters held constant:

    V dC/dt = Q (C_out - C) + S

Closed-form solution (used here as a sanity check on the ODE solver):

    C(t)  = C_ss + (C0 - C_ss) * exp(-Q t / V)
    C_ss  = C_out + S / Q
    tau   = V / Q                    (time constant)

Source convention:
    S_vol is volumetric CO2 generation in m^3 CO2 per hour.
    S in the ODE equals 1e6 * S_vol (so both sides of the ODE are in ppm * m^3 / h).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# --- Parameters --------------------------------------------------------------
V          = 100.0   # zone volume                 [m^3]
Q          = 500.0   # airflow (constant)          [m^3/h]    -> ACH = Q/V = 2 /h
S_vol      = 0.10    # CO2 generation (constant)   [m^3 CO2 / h]   ~5-6 sedentary adults
C_out      = 420.0   # outdoor CO2                 [ppm]
C0         = 500.0   # initial indoor CO2          [ppm]
duration_h = 8.0     # simulation length           [h]
dt_min     = 1.0     # output time step            [min]
sigma_meas = 15.0    # sensor noise (Gaussian)     [ppm]
seed       = 0

S = 1e6 * S_vol   # source term in ODE units [ppm * m^3 / h]
# this is unit conversion


# --- Simulate ----------------------------------------------------------------
n_steps = int(duration_h * 60 / dt_min) + 1
t_hours = np.linspace(0.0, duration_h, n_steps)

def rhs(t, C):
    return [(Q * (C_out - C[0]) + S) / V]

sol = solve_ivp(rhs, (t_hours[0], t_hours[-1]), [C0],
                t_eval=t_hours, method="RK45", rtol=1e-8, atol=1e-4)
C_true = sol.y[0] # exact simulated solution

# Closed-form reference
C_ss = C_out + S / Q
tau  = V / Q
C_analytic = C_ss + (C0 - C_ss) * np.exp(-Q * t_hours / V) # exact formula solution

# Noisy measurement
rng = np.random.default_rng(seed)
C_meas = C_true + rng.normal(0.0, sigma_meas, size=C_true.shape)


# --- Save --------------------------------------------------------------------
df = pd.DataFrame({
    "t_hours":        t_hours,
    "C_true_ppm":     C_true,
    "C_analytic_ppm": C_analytic,
    "C_meas_ppm":     C_meas,
})

OUT = Path("datasets")
OUT.mkdir(parents=True, exist_ok=True)
Q_int = int(Q)
csv_filename = f"iaq_co2_simple_Q{Q_int}.csv"
df.to_csv(OUT / csv_filename, index=False)

print(f"V = {V} m^3,  Q = {Q} m^3/h,  ACH = {Q/V:.2f} /h")
print(f"S_vol = {S_vol} m^3/h CO2,  C_out = {C_out} ppm,  C0 = {C0} ppm")
print(f"Steady-state C_ss = {C_ss:.1f} ppm")
print(f"Time constant tau = V/Q = {tau:.3f} h ({tau*60:.1f} min)")
print(f"Max |ODE - analytical| = {np.max(np.abs(C_true - C_analytic)):.2e} ppm")
print(f"Saved {len(df)} rows -> {OUT / csv_filename}")


# --- Quick plot --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(t_hours, C_meas, s=8, alpha=0.4, color="#888", label="C measured (noisy)")
ax.plot(t_hours, C_true, color="#c0392b", lw=2, label="C ODE")
ax.plot(t_hours, C_analytic, "--", color="black", lw=1.2, label="C analytical")
ax.axhline(C_ss, color="#2c7fb8", ls=":", lw=1, label=f"C_ss = {C_ss:.0f} ppm")
ax.set_xlabel("Time [h]")
ax.set_ylabel("CO₂ [ppm]")
ax.set_title(f"Constant S, constant Q  (V={V:.0f} m³, Q={Q_int} m³/h, τ={tau*60:.0f} min)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
png_filename = f"iaq_co2_simple_Q{Q_int}.png"
fig.savefig(OUT / png_filename, dpi=130, bbox_inches="tight")
print(f"Saved figure -> {OUT / png_filename}")
