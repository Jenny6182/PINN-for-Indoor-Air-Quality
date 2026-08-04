
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

from steady_state_algorithm.detect_transient_steady import label_steady, smoothed_derivative
from steady_state_algorithm.segment_cutter import cut_segments
from steady_state_algorithm.estimate_params import estimate_segment

def plot_true_vs_estimated_scatter(result):
    """
    Scatter plot: estimated vs true values.
    Perfect estimation lies on y=x.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))


    # Q scatter
    axes[0].scatter(
        result["Q_true"],
        result["Q_est"]
    )

    q_min = min(result["Q_true"].min(), result["Q_est"].min())
    q_max = max(result["Q_true"].max(), result["Q_est"].max())

    axes[0].plot(
        [q_min, q_max],
        [q_min, q_max]
    )

    axes[0].set_xlabel("Q true")
    axes[0].set_ylabel("Q estimate")
    axes[0].set_title("Q recovery")
    axes[0].grid()


    # S scatter
    axes[1].scatter(
        result["S_true"],
        result["S_est"]
    )

    s_min = min(result["S_true"].min(), result["S_est"].min())
    s_max = max(result["S_true"].max(), result["S_est"].max())

    axes[1].plot(
        [s_min, s_max],
        [s_min, s_max]
    )

    axes[1].set_xlabel("S true")
    axes[1].set_ylabel("S estimate")
    axes[1].set_title("S recovery")
    axes[1].grid()

    plt.tight_layout()
    plt.show()


def plot_true_vs_estimated_segments(df, t, C, segments,
                                    true_Q_col="Q_true",
                                    true_S_col="S_true"):
    """
    Compare true segment boundaries from dataset vs estimated boundaries.
    """

    plt.figure(figsize=(14, 5))

    # CO2 curve
    plt.plot(
        t,
        C,
        label="CO2 concentration"
    )

    # True segment boundaries
    true_Q = df[true_Q_col].to_numpy()
    true_S = df[true_S_col].to_numpy()

    true_change = np.where(
        (true_Q[1:] != true_Q[:-1]) |
        (true_S[1:] != true_S[:-1])
    )[0] + 1

    for i, idx in enumerate(true_change):
        plt.axvline(
            t[idx],
            linestyle="--",
            linewidth=2,
            label="True boundary" if i == 0 else None
        )


    # Estimated segment boundaries
    estimated_change = []

    for i, seg in enumerate(segments):
        if i == 0:
            continue

        estimated_change.append(seg.start_idx)

        plt.axvline(
            t[seg.start_idx],
            linestyle=":",
            linewidth=2,
            color="red",
            label="Estimated boundary" if i == 1 else None
        )

    estimated_change = np.array(estimated_change)

    offsets = []

    for true_idx, est_idx in zip(true_change, estimated_change):
        offsets.append(t[est_idx] - t[true_idx])

    
    print(offsets)


    plt.xlabel("Time (hours)")
    plt.ylabel("CO₂ (ppm)")
    plt.title("True vs Estimated Segment Boundaries")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_Q_true_with_boundaries(df, t, segments, true_Q_col="Q_true"):
    plt.figure(figsize=(14, 4))

    plt.plot(
        t,
        df[true_Q_col],
        label="Q_true"
    )

    # true Q changes
    Q = df[true_Q_col].to_numpy()

    true_change = np.where(Q[1:] != Q[:-1])[0] + 1

    for i, idx in enumerate(true_change):
        plt.axvline(
            t[idx],
            color="black",
            linestyle="--",
            label="True Q change" if i == 0 else None
        )

    # estimated boundaries
    for i, seg in enumerate(segments):
        if i == 0:
            continue

        plt.axvline(
            t[seg.start_idx],
            color="red",
            linestyle=":",
            label="Estimated boundary" if i == 1 else None
        )

    plt.xlabel("Time (hours)")
    plt.ylabel("Q")
    plt.title("True Q Changes vs Estimated Boundaries")
    plt.legend()
    plt.grid()
    plt.show()


def plot_derivative_with_Q(df, t, C, segments):
    dCdt = smoothed_derivative(t, C)

    fig, ax1 = plt.subplots(figsize=(14,4))

    ax1.plot(t, dCdt, label="dC/dt")
    ax1.set_ylabel("dC/dt")

    Q = df["Q_true"].to_numpy()

    ax2 = ax1.twinx()
    ax2.plot(t, Q, color="green", label="Q_true")
    ax2.set_ylabel("Q")

    for i, seg in enumerate(segments):
        if i != 0:
            ax1.axvline(
                t[seg.start_idx],
                color="red",
                linestyle=":"
            )

    plt.title("Derivative vs True Q Changes")
    plt.grid()
    plt.show()

def plot_css_debug(t, C, segments, result, true_css=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Full concentration curve
    ax.plot(t, C, label="C_meas")

    for i, seg in enumerate(segments):

        # Plot transient points
        if len(seg.transient_idx) > 0:
            ax.scatter(
                t[seg.transient_idx],
                C[seg.transient_idx],
                s=15,
                color="blue",
                label="transient" if i == 0 else None
            )

        # Plot steady points used for Css calculation
        if len(seg.steady_idx) > 0:
            ax.scatter(
                t[seg.steady_idx],
                C[seg.steady_idx],
                s=15,
                color="orange",
                label="steady points used for Css" if i == 0 else None
            )

        # Plot estimated Css
        if i < len(result):
            css = result.iloc[i]["Css_est"]
            if not np.isnan(css):
                ax.hlines(
                    css,
                    t[seg.start_idx],
                    t[seg.end_idx],
                    linestyles=":",
                    label="estimated Css" if i == 0 else None
                )

        # Plot true Css if provided
        if true_css is not None:
            ax.hlines(
                true_css[i],
                t[seg.start_idx],
                t[seg.end_idx],
                linestyles="--",
                label="true Css" if i == 0 else None
            )

        # Segment boundary
        ax.axvline(
            t[seg.start_idx],
            linestyle="--",
            alpha=0.5
        )

        ax.text(
            t[seg.start_idx],
            np.max(C),
            f"S{i}",
            rotation=90
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("CO₂ (ppm)")
    ax.set_title("Css Estimation Debug: Steady Points Used vs Estimated Css")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()



def compare_Css(result, C_out):
    """
    Compare estimated Css against true Css calculated from Q_true and S_true.
    """

    result = result.copy()

    result["Css_true"] = (
        C_out + result["S_true"] / result["Q_true"]
    )

    print(
        result[
            [
                "Css_true",
                "Css_est",
                "Q_true",
                "S_true"
            ]
        ].to_string(index=False)
    )

    plt.figure(figsize=(8, 4))

    plt.plot(
        range(len(result)),
        result["Css_true"],
        marker="o",
        label="True Css"
    )

    plt.plot(
        range(len(result)),
        result["Css_est"],
        marker="x",
        label="Estimated Css"
    )

    plt.xlabel("Segment")
    plt.ylabel("Css (ppm)")
    plt.title("True vs Estimated Css")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_tau_comparison(result, V):
    """
    Compare estimated tau against true tau from Q_true.
    """

    tau_true = V / result["Q_true"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(len(result)),
        tau_true,
        marker="o",
        label="True tau"
    )

    plt.plot(
        range(len(result)),
        result["tau_est"],
        marker="x",
        label="Estimated tau"
    )

    plt.xlabel("Segment")
    plt.ylabel("Tau (hours)")
    plt.title("True vs Estimated Tau")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_relative_error(result):
    """
    Plot relative estimation error (%) for Q and S.
    """

    segments = np.arange(len(result))

    Q_error = (
        (result["Q_est"] - result["Q_true"])
        / result["Q_true"]
        * 100
    )

    S_error = (
        (result["S_est"] - result["S_true"])
        / result["S_true"]
        * 100
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    width = 0.35

    ax.bar(
        segments - width/2,
        Q_error,
        width,
        label="Q error (%)"
    )

    ax.bar(
        segments + width/2,
        S_error,
        width,
        label="S error (%)"
    )

    ax.axhline(0)

    ax.set_xlabel("Segment")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative Parameter Estimation Error")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()


def plot_true_vs_estimated_with_error(result):
    """
    True vs estimated Q and S with percentage error annotations.
    """

    segments = np.arange(len(result))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q plot
    axes[0].plot(
        segments,
        result["Q_true"],
        "o-",
        label="Q true"
    )

    axes[0].plot(
        segments,
        result["Q_est"],
        "x--",
        label="Q estimate"
    )

    for i in segments:
        error = (
            (result["Q_est"].iloc[i] - result["Q_true"].iloc[i])
            / result["Q_true"].iloc[i]
            * 100
        )

        axes[0].text(
            i,
            result["Q_est"].iloc[i],
            f"{error:.1f}%",
            fontsize=9
        )

    axes[0].set_title("Q: True vs Estimated")
    axes[0].set_xlabel("Segment")
    axes[0].set_ylabel("Q")
    axes[0].legend()
    axes[0].grid()


    # S plot
    axes[1].plot(
        segments,
        result["S_true"],
        "o-",
        label="S true"
    )

    axes[1].plot(
        segments,
        result["S_est"],
        "x--",
        label="S estimate"
    )

    for i in segments:
        error = (
            (result["S_est"].iloc[i] - result["S_true"].iloc[i])
            / result["S_true"].iloc[i]
            * 100
        )

        axes[1].text(
            i,
            result["S_est"].iloc[i],
            f"{error:.1f}%",
            fontsize=9
        )

    # zoom y-axis so differences are visible but not exaggerated
    s_min = min(
        result["S_true"].min(),
        result["S_est"].min()
    )

    s_max = max(
        result["S_true"].max(),
        result["S_est"].max()
    )

    margin = (s_max - s_min) * 0.5

    axes[1].set_ylim(
        s_min - margin,
        s_max + margin
    )

    axes[1].set_title("S: True vs Estimated (Zoomed)")
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("S")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()