"""
Simple IAQ Scenario — Constant Source, Constant Ventilation
----------------------------------------------------------
Single well-mixed zone:

    V dC/dt = Q (C_out - C) + S

Analytical solution:

    C(t) = C_ss + (C0 - C_ss) exp(-Qt/V)
    C_ss = C_out + S/Q
    tau  = V/Q
"""

from dataclasses import dataclass
from core_data_generator import *

@dataclass
class ConstantConfig:
    Q: float = 500.0
    S_vol: float = 0.10
    seed: int = 0
    phys: PhysicalParams = field(default_factory=PhysicalParams)
    out_dir: str = "datasets"
    save_csv: bool = True
    save_plot: bool = True

def run(cfg: ConstantConfig) -> dict:
    S = cfg.S_vol * 1e6
    t = build_time_grid(cfg.phys)
    C_true = solve_ode(lambda _: cfg.Q, lambda _: S, t, cfg.phys)
    C_ss = cfg.phys.C_out + S / cfg.Q
    C_analytic = C_ss + (cfg.phys.C0 - C_ss) * np.exp(-cfg.Q * t / cfg.phys.V)
    C_meas = add_noise(C_true, cfg.phys.sigma_meas, cfg.seed)

    # --- build df ---
    df = pd.DataFrame({
        "t_hours":       t,
        "C_true_ppm":    C_true,
        "C_analytic_ppm": C_analytic,
        "C_meas_ppm":    C_meas,
    })

    OUT = Path(cfg.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"iaq_Q{int(cfg.Q)}_S{cfg.S_vol:.2f}"

    # --- save csv ---
    if cfg.save_csv:
        save_csv(df, OUT / f"{stem}.csv")

    # --- plot ---
    if cfg.save_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(t, C_meas, s=8, alpha=0.4, color="#888", label="Measured")
        ax.plot(t, C_true, color="#c0392b", lw=2, label="ODE")
        ax.plot(t, C_analytic, "--", color="black", lw=1.2, label="Analytical")
        ax.axhline(C_ss, color="#2c7fb8", ls=":", label=f"C_ss={C_ss:.0f}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("CO₂ [ppm]")
        ax.set_title(f"Q={cfg.Q:.0f}, S_vol={cfg.S_vol:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_plot(fig, OUT / f"{stem}.png")

    return {
        "df": df,
        "C_true": C_true,
        "C_analytic": C_analytic,
        "C_meas": C_meas,
        "C_ss": C_ss,
    }