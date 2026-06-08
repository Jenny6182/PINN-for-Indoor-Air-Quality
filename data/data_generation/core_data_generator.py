"""core_data_generator.py includes all the helper functions required to create synthesized data"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

@dataclass
class PhysicalParams:
    V: float = 100.0
    C_out: float = 420.0
    C0: float = 500.0
    duration_h: float = 8.0
    dt_min: float = 1.0
    sigma_meas: float = 15.0

def build_time_grid(p: PhysicalParams):
    n = int(p.duration_h * 60 / p.dt_min) + 1
    return np.linspace(0.0, p.duration_h, n)

def solve_ode(Q_func, S_func, t_hours, p: PhysicalParams):
    def rhs(t, C):
        return [(Q_func(t) * (p.C_out - C[0]) + S_func(t)) / p.V]
    sol = solve_ivp(rhs, (t_hours[0], t_hours[-1]), [p.C0],
                    t_eval=t_hours, method="RK45", rtol=1e-8, atol=1e-4)
    return sol.y[0]

def add_noise(C_true, sigma, seed):
    return C_true + np.random.default_rng(seed).normal(0.0, sigma, C_true.shape)

def make_step_func(values, seg_dur, n_seg):
    def f(t):
        return float(values[min(int(t / seg_dur), n_seg - 1)])
    return f

def save_csv(df, path: Path):
    df.to_csv(path, index=False)

def save_plot(fig, path: Path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)