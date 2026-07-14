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
from dataclasses import dataclass, field
from core_data_generator import *

@dataclass
class VaryingConfig:
    mode: str = "Q" # "Q" or "S"
    seed: int = 42
    noise_seed: int = 0
    max_segments: int = 15
    Q_step_max: float = 100      # normal change between segments
    Q_jump_prob: float = 0.15    # probability of HVAC mode switch
    Q_range: tuple = (200, 700) # used when mode="Q"
    S_range: tuple = (0.03, 0.70) # used when mode="S"
    Q_const: float = 200.0 # used when mode="S"
    S_const: float = 0.10 # used when mode="Q"
    phys: PhysicalParams = field(default_factory=PhysicalParams)
    out_dir: str = "datasets"
    save_csv: bool = True
    save_plot: bool = True

def run(cfg: VaryingConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    
    # 1. Enforce the physics constraint (0.5 hours = 30 mins)
    min_step_h = 0.5 
    
    # Calculate absolute max possible segments so we don't break the math
    # (e.g., if duration is 8 hours, max possible is 16 segments)
    max_possible_segments = int(cfg.phys.duration_h / min_step_h)
    # Instead of rng.integers(5, 21), you now have a flexible limit
    upper_bound = min(cfg.max_segments, max_possible_segments + 1)
    
    # Pick a random number of segments (e.g., between 5 and the safe upper bound)
    actual_n_segments = rng.integers(5, upper_bound)

    # 2. Distribute the time using the "leftover" method
    # Give every segment its mandatory 30 mins, calculate what is left over
    leftover_time = cfg.phys.duration_h - (actual_n_segments * min_step_h)

    # Generate random proportions to divide up the leftover time
    random_props = rng.uniform(0, 1, actual_n_segments)
    random_props /= np.sum(random_props) # Normalize so they equal exactly 1.0

    # Add the mandatory 30 mins to the random leftover chunks
    segment_durations = min_step_h + (random_props * leftover_time)

    # 3. Build the timeline boundaries using a cumulative sum
    # (Drop the final sum because it equals duration_h, which we append manually)
    changepoints = np.cumsum(segment_durations)[:-1]
    
    boundaries = np.concatenate(([0], changepoints, [cfg.phys.duration_h]))

    if cfg.mode == "Q":
        Q_min, Q_max = cfg.Q_range

        Q_vals = np.zeros(actual_n_segments)

        # Initial ventilation level
        Q_vals[0] = rng.uniform(Q_min, Q_max)

        for i in range(1, actual_n_segments):

            # Occasionally switch operating mode
            # (e.g. occupied -> unoccupied, schedule change)
            if rng.random() < 0.15:
                Q_vals[i] = rng.uniform(Q_min, Q_max)

            else:
                # Gradual HVAC adjustment
                delta = rng.uniform(-100, 100)
                Q_vals[i] = Q_vals[i-1] + delta

                # Keep physically reasonable
                Q_vals[i] = np.clip(Q_vals[i], Q_min, Q_max)

        Q_func = make_variable_step_func(Q_vals, boundaries)

        S_func = lambda _: cfg.S_const * 1e6
        label = f"varyQ_seg{actual_n_segments}_seed{cfg.seed}"


    else:
        S_min, S_max = cfg.S_range

        S_vals = np.zeros(actual_n_segments)

        # Initial occupancy level
        S_vals[0] = rng.uniform(S_min, S_max)

        for i in range(1, actual_n_segments):

            # People entering/leaving causes jumps
            if rng.random() < 0.30:
                S_vals[i] = rng.uniform(S_min, S_max)

            else:
                # Occupancy stays roughly similar
                delta = rng.uniform(-0.05, 0.05)
                S_vals[i] = S_vals[i-1] + delta

                # Keep within range
                S_vals[i] = np.clip(S_vals[i], S_min, S_max)

        Q_func = lambda _: cfg.Q_const

        S_func = make_variable_step_func(
            S_vals * 1e6,
            boundaries
        )

        label = f"varyS_seg{actual_n_segments}_seed{cfg.seed}"

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
    
    np.savez(OUT / f"iaq_{label}_truth.npz",
         boundaries=boundaries,          # changepoint locations
         values=Q_vals if cfg.mode=="Q" else S_vals,
         mode=cfg.mode,
         n_segments=actual_n_segments)

    return {"df": df, "C_true": C_true, "C_meas": C_meas,
            "Q_arr": Q_arr, "S_arr": S_arr, "n_segments": actual_n_segments}


seeds = [87, 14, 51, 3, 96, 68, 39, 77, 22, 90, 45, 12, 83, 57, 26, 71, 9, 64, 35, 92]

for i in seeds:
    cfg = VaryingConfig(seed=i)
    run(cfg)

# cfg = VaryingConfig(seed=2)
# run(cfg)