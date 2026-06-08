"""
IAQ CO2 Piecewise Data Generator
=================================

Generates two datasets with piecewise-constant parameters:

    V dC/dt = Q(t) * (C_out - C) + S(t)

Scenario A, varying_Q: Q changes every 0.5h, S is constant
Scenario B, varying_S: S changes every 0.5h, Q is constant

Outputs (saved to ./varying_datasets/):
    iaq_co2_varying_Q.csv + iaq_co2_varying_Q.png
    iaq_co2_varying_S.csv + iaq_co2_varying_S.png

Each CSV has columns:
    t_hours, C_true_ppm, C_meas_ppm, Q_true, S_true (Q or S is constant for each range of time)

Note: no closed-form analytical solution exists for piecewise Q/S,
so only the ODE solution (solve_ivp) is used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# --------- fixed physical constants-------
V          = 100.0   # zone volume          [m^3]
C_out      = 420.0   # outdoor CO2          [ppm]
C0         = 500.0   # initial indoor CO2   [ppm]
duration_h = 8.0     # simulation length    [h]
dt_min     = 1.0     # output time step     [min]
sigma_meas = 15.0    # sensor noise (std)   [ppm]
seed       = 0

# --------- piecewise schedule ---------
segment_duration = 0.5                              # each segment lasts 0.5 h
n_segments       = int(duration_h / segment_duration)  # = 16 segments

rng = np.random.default_rng(seed=42)

# scenario A: Q varies, S constant
Q_values_A = rng.uniform(100, 800, size=n_segments)  # 16 random Q values
S_vol_A    = 0.10                                    # constant ~5-6 people

# scenario B: Q constant, S varies
Q_B            = 200.0                                       # constant ventilation
S_vol_values_B = rng.uniform(0.03, 0.70, size=n_segments)   # 16 random S_vol values


# --------- helper: make piecewise-constant function from an array of segment values -----------
def make_step_func(values, seg_dur, n_seg):
    """Returns f(t) that looks up the correct segment value for time t."""
    def func(t):
        idx = min(int(t / seg_dur), n_seg - 1)
        return float(values[idx])
    return func

Q_func_A = make_step_func(Q_values_A, segment_duration, n_segments)
S_func_A = lambda t: S_vol_A * 1e6 # constant, converted to ppm·m³/h

Q_func_B = lambda t: Q_B # constant
S_func_B = make_step_func(S_vol_values_B*1e6, segment_duration, n_segments)



def extract_schedule(values_Q, values_S, segment_duration, n_segments,
                     Q_const=None, S_const=None, mode="varying_Q"):
    """
    Returns:
        change_points: list of times [0.5, 1.0, 1.5, ...]
        Q_list: per-segment Q values (or constant repeated)
        S_list: per-segment S values (or constant repeated)
        Q_constant / S_constant: scalar constants when applicable
    """
    change_points = [(i + 1) * segment_duration for i in range(n_segments)]

    if mode == "varying_Q":
        Q_list = list(values_Q)
        S_list = [S_const] * n_segments
        return change_points, Q_list, S_list, S_const

    elif mode == "varying_S":
        Q_list = [Q_const] * n_segments
        S_list = list(values_S)
        return change_points, Q_list, S_list, Q_const

    else:
        raise ValueError("mode must be 'varying_Q' or 'varying_S'")


# ----------- simulate and save ----------
def simulate_and_save(Q_func, S_func, label, OUT):
    """
    Integrate the ODE, add noise, save CSV and diagnostic plot.

    Parameters
    ----------
    Q_func : callable  t -> Q [m^3/h]
    S_func : callable  t -> S [ppm * m^3/h]
    label  : str       used in filenames, e.g. 'varying_Q'
    OUT    : Path      output directory
    """
    n_steps = int(duration_h * 60 / dt_min) + 1
    t_hours = np.linspace(0.0, duration_h, n_steps)

    # --- ODE ---
    def rhs(t, C):
        return [(Q_func(t) * (C_out - C[0]) + S_func(t)) / V]

    sol = solve_ivp(
        rhs, (t_hours[0], t_hours[-1]), [C0],
        t_eval=t_hours, method="RK45", rtol=1e-8, atol=1e-4,
    )
    C_true = sol.y[0]

    # evaluate Q and S at every output time (for saving and plotting)
    Q_arr = np.array([Q_func(t) for t in t_hours])
    S_arr = np.array([S_func(t) for t in t_hours])

    # --- noisy measurement ---
    rng_noise = np.random.default_rng(seed)
    C_meas = C_true + rng_noise.normal(0.0, sigma_meas, size=C_true.shape)

    # --- save CSV ---
    df = pd.DataFrame({
        "t_hours":    t_hours,
        "C_true_ppm": C_true,
        "C_meas_ppm": C_meas,
        "Q_true":     Q_arr,           # ground truth Q at each timestep
        "S_true":     S_arr,           # ground truth S at each timestep
    })
    csv_path = OUT / f"iaq_co2_{label}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}  ({len(df)} rows)")

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # top panel: CO2 concentration
    ax1.scatter(t_hours, C_meas,  s=8,  alpha=0.4, color="#888",    label="C meas (noisy)")
    ax1.plot(   t_hours, C_true,  lw=2, color="#c0392b",             label="C true (ODE)")
    ax1.set_ylabel("CO₂ [ppm]")
    ax1.set_title(f"IAQ simulation — {label}  (V={V:.0f} m³, σ={sigma_meas} ppm)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # bottom panel: parameter schedule
    # bottom panel: only show varying parameter

    if label == "varying_Q":
        ax2.step(
            t_hours,
            Q_arr,
            color="#2980b9",
            lw=2,
            where="post"
        )
        ax2.set_ylabel("Q [m³/h]")
        ax2.text(
            0.02, 0.95,
            f"S_vol = {S_vol_A:.2f} m³ CO₂/h",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8)
        )

    elif label == "varying_S":
        ax2.step(
            t_hours,
            S_arr / 1e6,
            color="#e67e22",
            lw=2,
            where="post"
        )
        ax2.set_ylabel("S_vol [m³ CO₂/h]")
        ax2.text(
            0.02, 0.95,
            f"Q = {Q_B:.0f} m³/h",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8)
        )

    ax2.set_xlabel("Time [h]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = OUT / f"iaq_co2_{label}.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"Saved {png_path}")

    tau_list = []
    Q_list = []
    S_list = []

    # print segment summary
    print(f"\n  Segment values for {label}:")
    print(f"  {'seg':>4}  {'t_start':>8}  {'t_end':>6}  {'Q':>8}  {'S_vol':>8}")
    for i in range(n_segments):
        t_start = i * segment_duration
        tau_list.append(t_start)
        t_end   = (i + 1) * segment_duration
        q_val   = Q_func(t_start)
        Q_list.append(q_val)
        s_val   = S_func(t_start) / 1e6
        S_list.append(s_val)
        print(f"  {i:>4}  {t_start:>8.2f}  {t_end:>6.2f}  {q_val:>8.1f}  {s_val:>8.4f}")

    print()

    print("changepoints: " , tau_list)
    print("Q_list: ", Q_list)
    print("S_list: ", S_list)


# run both scenarios
# ---------------------------------------------------------------------------
OUT = Path("datasets")
OUT.mkdir(parents=True, exist_ok=True)

simulate_and_save(Q_func_A, S_func_A, "varying_Q", OUT)
simulate_and_save(Q_func_B, S_func_B, "varying_S", OUT)

plt.show()