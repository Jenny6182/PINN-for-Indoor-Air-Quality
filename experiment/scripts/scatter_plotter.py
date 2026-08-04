"""
Generate Q-S parameter recovery scatter plots from previously trained ensemble runs.

This script assumes that each ensemble seed has already been trained and that
the corresponding history.json files exist. It reconstructs the final Q(t) and
S(t) estimates from the learned changepoints and segment values, then compares
estimated parameter states against the true states from the validation dataset.

Generates three plot variants per dataset:
    1. Original      - all seeds flattened together, single color
    2. By true segment - colored by which TRUE segment (from ground truth) each
                          point's timestamp falls in. NOT colored by each seed's
                          own detected segments, since those aren't comparable
                          across seeds (different seeds can detect different
                          numbers/positions of changepoints).
    3. By seed        - each seed gets its own color, to spot outlier seeds.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from core.utils.reconstruction import reconstruct_step_function


DATA_ROOT = Path("results/ensemble_runs")
OUTPUT_DIR = Path("results/ensemble_plots/QS_scatter")
RAW_DATA_DIR = Path("data/datasets/validation_dataset")

N_ENSEMBLE = 20

TARGET_DATASETS = {
    "Easy": "iaq_varyQ_seg11_seed12.csv",
    "Average": "iaq_varyQ_seg10_seed64.csv",
    "Hard": "iaq_varyQ_seg8_seed51.csv"
}


def true_segment_ids(Q_true: np.ndarray, S_true: np.ndarray) -> np.ndarray:
    """
    Assign each timepoint a segment index based on the TRUE (Q,S) state,
    not any individual seed's detected changepoints. This is what makes
    coloring meaningful across seeds -- every seed shares the same
    t_eval grid, so "true segment 3" means the same thing for all of them.
    """
    changes = (np.diff(Q_true) != 0) | (np.diff(S_true) != 0)
    seg_id = np.zeros(len(Q_true), dtype=int)
    seg_id[1:] = np.cumsum(changes)
    return seg_id


def load_estimates(difficulty: str, t_eval: np.ndarray):
    """
    Load per-seed Q/S reconstructions. Returns:
        seeds_used   : list of seed numbers that had a history.json
        Q_estimates  : list of arrays, one per seed, each shape (len(t_eval),)
        S_estimates  : list of arrays, one per seed
    """
    seeds_used = []
    Q_estimates, S_estimates = [], []

    for seed in range(1, N_ENSEMBLE + 1):
        history_path = DATA_ROOT / f"{difficulty}_seed{seed}" / "history.json"

        if not history_path.exists():
            print(f"Missing {history_path}")
            continue

        with open(history_path) as f:
            history = json.load(f)

        taus = np.array(history["taus"])[-1]
        Q_values = np.array(history["Q"])[-1]
        S_values = np.array(history["S"])[-1]

        seeds_used.append(seed)
        Q_estimates.append(reconstruct_step_function(t_eval, taus, Q_values))
        S_estimates.append(reconstruct_step_function(t_eval, taus, S_values))

    return seeds_used, Q_estimates, S_estimates


def plot_original(S_flat, Q_flat, true_states, difficulty, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(S_flat, Q_flat, s=12, alpha=0.15, label="Estimated")
    ax.scatter(
        true_states[:, 0], true_states[:, 1],
        s=150, marker="*", color="red", edgecolor="black", label="True"
    )

    ax.set_xlabel("Source Rate S")
    ax.set_ylabel("Ventilation Rate Q")
    ax.set_title(f"{difficulty}: Q-S Parameter Recovery")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_by_true_segment(S_flat, Q_flat, seg_id_flat, true_states, difficulty, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))

    n_segments = seg_id_flat.max() + 1
    cmap = cm.get_cmap("viridis", n_segments)

    sc = ax.scatter(
        S_flat, Q_flat,
        c=seg_id_flat, cmap=cmap, s=12, alpha=0.25,
        vmin=-0.5, vmax=n_segments - 0.5,
    )
    ax.scatter(
        true_states[:, 0], true_states[:, 1],
        s=150, marker="*", color="red", edgecolor="black", label="True", zorder=5
    )

    cbar = plt.colorbar(sc, ax=ax, ticks=range(n_segments))
    cbar.set_label("True segment index")

    ax.set_xlabel("Source Rate S")
    ax.set_ylabel("Ventilation Rate Q")
    ax.set_title(f"{difficulty}: Q-S Recovery by True Segment")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_by_seed(seeds_used, Q_estimates, S_estimates, true_states, difficulty, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))

    n_seeds = len(seeds_used)
    cmap = cm.get_cmap("tab20", n_seeds)

    for i, (seed, Q_seed, S_seed) in enumerate(zip(seeds_used, Q_estimates, S_estimates)):
        ax.scatter(
            S_seed, Q_seed,
            s=10, alpha=0.35, color=cmap(i),
            label=f"seed {seed}" if n_seeds <= 20 else None,
        )

    ax.scatter(
        true_states[:, 0], true_states[:, 1],
        s=150, marker="*", color="red", edgecolor="black", label="True", zorder=5
    )

    ax.set_xlabel("Source Rate S")
    ax.set_ylabel("Ventilation Rate Q")
    ax.set_title(f"{difficulty}: Q-S Recovery by Seed")
    ax.grid(alpha=0.3)
    # legend gets crowded with 20 seeds -- put it outside the axes
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for difficulty, filename in TARGET_DATASETS.items():
        print(f"Plotting {difficulty}...")

        df = pd.read_csv(RAW_DATA_DIR / filename)

        t_eval = df["t_hours"].values
        Q_true = df["Q_true"].values
        S_true = df["S_true"].values

        true_states = np.unique(np.column_stack([S_true, Q_true]), axis=0)
        seg_id = true_segment_ids(Q_true, S_true)  # shape (len(t_eval),)

        seeds_used, Q_estimates, S_estimates = load_estimates(difficulty, t_eval)

        if not Q_estimates:
            print("No predictions found.")
            continue

        Q_arr = np.array(Q_estimates)  # (n_seeds, len(t_eval))
        S_arr = np.array(S_estimates)

        Q_flat = Q_arr.flatten()
        S_flat = S_arr.flatten()
        # tile seg_id once per seed so it lines up with the flattened arrays
        seg_id_flat = np.tile(seg_id, len(seeds_used))

        # --- 1. original ---
        plot_original(
            S_flat, Q_flat, true_states, difficulty,
            OUTPUT_DIR / f"{difficulty}_QS_scatter.png",
        )

        # --- 2. colored by true segment ---
        plot_by_true_segment(
            S_flat, Q_flat, seg_id_flat, true_states, difficulty,
            OUTPUT_DIR / f"{difficulty}_QS_scatter_by_segment.png",
        )

        # --- 3. colored by seed ---
        plot_by_seed(
            seeds_used, Q_estimates, S_estimates, true_states, difficulty,
            OUTPUT_DIR / f"{difficulty}_QS_scatter_by_seed.png",
        )


if __name__ == "__main__":
    main()