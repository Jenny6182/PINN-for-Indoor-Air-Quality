"""
Simple IAQ Scenario — Piecewise Constant Source/Ventilation
----------------------------------------------------------
Single well-mixed zone:

    V dC/dt = Q (C_out - C) + S

Analytical solution for each segment:

    C(t) = C_ss + (C0 - C_ss) exp(-Qt/V)
    C_ss = C_out + S/Q
    tau  = V/Q
"""


# varying.py
from dataclasses import dataclass, field
from core_data_generator import *

@dataclass
class VaryingConfig:
    mode: str = "Q" # "Q" or "S"
    seed: int = 42
    noise_seed: int = 0
    n_segments: int = 16
    Q_range: tuple = (100, 800) # used when mode="Q"
    S_range: tuple = (0.03, 0.70) # used when mode="S"
    Q_const: float = 200.0 # used when mode="S"
    S_const: float = 0.10 # used when mode="Q"
    phys: PhysicalParams = field(default_factory=PhysicalParams)
    out_dir: str = "datasets"
    save_csv: bool = True
    save_plot: bool = True

def run(cfg: VaryingConfig) -> dict:
    seg_dur = cfg.phys.duration_h / cfg.n_segments
    rng = np.random.default_rng(cfg.seed)

    if cfg.mode == "Q":
        Q_vals = rng.uniform(*cfg.Q_range, cfg.n_segments)
        Q_func = make_step_func(Q_vals, seg_dur, cfg.n_segments)
        S_func = lambda _: cfg.S_const * 1e6
        label  = f"varyQ_seg{cfg.n_segments}_seed{cfg.seed}"
    else:
        S_vals = rng.uniform(*cfg.S_range, cfg.n_segments)
        Q_func = lambda _: cfg.Q_const
        S_func = make_step_func(S_vals * 1e6, seg_dur, cfg.n_segments)
        label  = f"varyS_seg{cfg.n_segments}_seed{cfg.seed}"

    t = build_time_grid(cfg.phys)
    C_true = solve_ode(Q_func, S_func, t, cfg.phys)
    C_meas = add_noise(C_true, cfg.phys.sigma_meas, cfg.noise_seed)

    # evaluate schedules at every timestep for saving
    Q_arr = np.array([Q_func(ti) for ti in t])
    S_arr = np.array([S_func(ti) for ti in t])

    df = pd.DataFrame({
        "t_hours":    t,
        "C_true_ppm": C_true,
        "C_meas_ppm": C_meas,
        "Q_true":     Q_arr,
        "S_true":     S_arr,
    })

    OUT = Path(cfg.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    if cfg.save_csv:
        save_csv(df, OUT / f"iaq_{label}.csv")

    if cfg.save_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.scatter(t, C_meas, s=8, alpha=0.4, color="#888", label="C meas")
        ax1.plot(t, C_true, lw=2, color="#c0392b", label="C true")
        ax1.set_ylabel("CO₂ [ppm]")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        if cfg.mode == "Q":
            ax2.step(t, Q_arr, color="#2980b9", lw=2, where="post")
            ax2.set_ylabel("Q [m³/h]")
            ax2.text(0.02, 0.95, f"S_const={cfg.S_const:.2f} m³/h",
                     transform=ax2.transAxes, va="top",
                     bbox=dict(facecolor="white", alpha=0.8))
        else:
            ax2.step(t, S_arr / 1e6, color="#e67e22", lw=2, where="post")
            ax2.set_ylabel("S_vol [m³/h]")
            ax2.text(0.02, 0.95, f"Q_const={cfg.Q_const:.0f} m³/h",
                     transform=ax2.transAxes, va="top",
                     bbox=dict(facecolor="white", alpha=0.8))

        ax2.set_xlabel("Time [h]")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        save_plot(fig, OUT / f"iaq_{label}.png")

    return {"df": df, "C_true": C_true, "C_meas": C_meas,
            "Q_arr": Q_arr, "S_arr": S_arr}