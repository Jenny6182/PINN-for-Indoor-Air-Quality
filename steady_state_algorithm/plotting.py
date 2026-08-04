import matplotlib.pyplot as plt
import numpy as np

from steady_state_algorithm.detect_transient_steady import smoothed_derivative


def plot_true_vs_estimated_scatter(result):
    """Scatter plot of estimated vs true values. Perfect estimation lies on y=x."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q recovery
    axes[0].scatter(result["Q_true"], result["Q_est"])
    q_min = min(result["Q_true"].min(), result["Q_est"].min())
    q_max = max(result["Q_true"].max(), result["Q_est"].max())
    axes[0].plot([q_min, q_max], [q_min, q_max])
    axes[0].set(xlabel="Q true", ylabel="Q estimate", title="Q recovery")
    axes[0].grid()

    # S recovery
    axes[1].scatter(result["S_true"], result["S_est"])
    s_min = min(result["S_true"].min(), result["S_est"].min())
    s_max = max(result["S_true"].max(), result["S_est"].max())
    axes[1].plot([s_min, s_max], [s_min, s_max])
    axes[1].set(xlabel="S true", ylabel="S estimate", title="S recovery")
    axes[1].grid()

    plt.tight_layout()
    plt.show()


def plot_true_vs_estimated_segments(df, t, C, segments, true_Q_col="Q_true", true_S_col="S_true"):
    """Compare true segment boundaries against estimated boundaries."""

    plt.figure(figsize=(14, 5))
    plt.plot(t, C, label="CO2 concentration")

    # Find true parameter changes
    true_Q = df[true_Q_col].to_numpy()
    true_S = df[true_S_col].to_numpy()
    true_change = np.where((true_Q[1:] != true_Q[:-1]) | (true_S[1:] != true_S[:-1]))[0] + 1

    for i, idx in enumerate(true_change):
        plt.axvline(t[idx], linestyle="--", linewidth=2, label="True boundary" if i == 0 else None)

    # Find estimated segment boundaries
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
            label="Estimated boundary" if i == 1 else None,
        )

    offsets = [t[e] - t[tr] for tr, e in zip(true_change, estimated_change)]
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

    plt.plot(t, df[true_Q_col], label="Q_true")

    Q = df[true_Q_col].to_numpy()
    true_change = np.where(Q[1:] != Q[:-1])[0] + 1

    # True Q changes
    for i, idx in enumerate(true_change):
        plt.axvline(t[idx], color="black", linestyle="--", label="True Q change" if i == 0 else None)

    # Estimated boundaries
    for i, seg in enumerate(segments):
        if i != 0:
            plt.axvline(t[seg.start_idx], color="red", linestyle=":", label="Estimated boundary" if i == 1 else None)

    plt.xlabel("Time (hours)")
    plt.ylabel("Q")
    plt.title("True Q Changes vs Estimated Boundaries")
    plt.legend()
    plt.grid()
    plt.show()


def plot_derivative_with_Q(df, t, C, segments):
    dCdt = smoothed_derivative(t, C)

    fig, ax1 = plt.subplots(figsize=(14, 4))

    ax1.plot(t, dCdt, label="dC/dt")
    ax1.set_ylabel("dC/dt")

    ax2 = ax1.twinx()
    ax2.plot(t, df["Q_true"], color="green", label="Q_true")
    ax2.set_ylabel("Q")

    for i, seg in enumerate(segments):
        if i != 0:
            ax1.axvline(t[seg.start_idx], color="red", linestyle=":")

    plt.title("Derivative vs True Q Changes")
    plt.grid()
    plt.show()


def plot_css_debug(t, C, segments, result, true_css=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t, C, label="C_meas")

    for i, seg in enumerate(segments):
        # Points classified as transient
        if len(seg.transient_idx) > 0:
            ax.scatter(
                t[seg.transient_idx],
                C[seg.transient_idx],
                s=15,
                color="blue",
                label="transient" if i == 0 else None,
            )

        # Points used for Css estimation
        if len(seg.steady_idx) > 0:
            ax.scatter(
                t[seg.steady_idx],
                C[seg.steady_idx],
                s=15,
                color="orange",
                label="steady points used for Css" if i == 0 else None,
            )

        # Estimated Css
        if i < len(result):
            css = result.iloc[i]["Css_est"]
            if not np.isnan(css):
                ax.hlines(css, t[seg.start_idx], t[seg.end_idx], linestyles=":", label="estimated Css" if i == 0 else None)

        # True Css
        if true_css is not None:
            ax.hlines(true_css[i], t[seg.start_idx], t[seg.end_idx], linestyles="--", label="true Css" if i == 0 else None)

        ax.axvline(t[seg.start_idx], linestyle="--", alpha=0.5)
        ax.text(t[seg.start_idx], np.max(C), f"S{i}", rotation=90)

    ax.set(
        xlabel="Time (hours)",
        ylabel="CO₂ (ppm)",
        title="Css Estimation Debug: Steady Points Used vs Estimated Css",
    )
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()


def compare_Css(result, C_out):
    """Compare estimated Css against true Css calculated from Q_true and S_true."""

    result = result.copy()
    result["Css_true"] = C_out + result["S_true"] / result["Q_true"]

    print(result[["Css_true", "Css_est", "Q_true", "S_true"]].to_string(index=False))

    plt.figure(figsize=(8, 4))

    plt.plot(range(len(result)), result["Css_true"], marker="o", label="True Css")
    plt.plot(range(len(result)), result["Css_est"], marker="x", label="Estimated Css")

    plt.xlabel("Segment")
    plt.ylabel("Css (ppm)")
    plt.title("True vs Estimated Css")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_tau_comparison(result, V):
    """Compare estimated tau against true tau calculated from Q_true."""

    tau_true = V / result["Q_true"]

    plt.figure(figsize=(8, 5))

    plt.plot(range(len(result)), tau_true, marker="o", label="True tau")
    plt.plot(range(len(result)), result["tau_est"], marker="x", label="Estimated tau")

    plt.xlabel("Segment")
    plt.ylabel("Tau (hours)")
    plt.title("True vs Estimated Tau")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_relative_error(result):
    """Plot relative estimation error (%) for Q and S."""

    segments = np.arange(len(result))

    Q_error = (result["Q_est"] - result["Q_true"]) / result["Q_true"] * 100
    S_error = (result["S_est"] - result["S_true"]) / result["S_true"] * 100

    fig, ax = plt.subplots(figsize=(10, 4))

    width = 0.35

    ax.bar(segments - width / 2, Q_error, width, label="Q error (%)")
    ax.bar(segments + width / 2, S_error, width, label="S error (%)")

    ax.axhline(0)

    ax.set(
        xlabel="Segment",
        ylabel="Relative error (%)",
        title="Relative Parameter Estimation Error",
    )

    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()


def plot_parameter_with_error(ax, segments, true_values, estimated_values, name):
    """Plot true vs estimated parameter with percentage error annotations."""

    ax.plot(segments, true_values, "o-", label=f"{name} true")
    ax.plot(segments, estimated_values, "x--", label=f"{name} estimate")

    for i in segments:
        error = (estimated_values.iloc[i] - true_values.iloc[i]) / true_values.iloc[i] * 100
        ax.text(i, estimated_values.iloc[i], f"{error:.1f}%", fontsize=9)

    ax.set_xlabel("Segment")
    ax.set_ylabel(name)
    ax.set_title(f"{name}: True vs Estimated")
    ax.legend()
    ax.grid()


def plot_true_vs_estimated_with_error(result):
    """
    True vs estimated Q and S with percentage error annotations.
    """

    segments = np.arange(len(result))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_parameter_with_error(
        axes[0],
        segments,
        result["Q_true"],
        result["Q_est"],
        "Q",
    )

    plot_parameter_with_error(
        axes[1],
        segments,
        result["S_true"],
        result["S_est"],
        "S",
    )

    # Zoom S axis because small changes are difficult to see
    s_min = min(result["S_true"].min(), result["S_est"].min())
    s_max = max(result["S_true"].max(), result["S_est"].max())
    margin = (s_max - s_min) * 0.5

    axes[1].set_ylim(s_min - margin, s_max + margin)
    axes[1].set_title("S: True vs Estimated (Zoomed)")

    plt.tight_layout()
    plt.show()


def plot_stage_comparison(t, C, stages, true_change_times=None):
    """
    Compare steady/transient boolean labels across multiple pipeline
    stages, stacked vertically, sharing the same time axis.

    stages: dict of {stage_name: steady_boolean_array}, e.g.
        {
            "raw t-stat (pre-filter)": raw_transient_as_steady_bool,
            "label_steady_significance_v2": steady,
            "after cut_segments/refine": steady_from_segments,
        }
    true_change_times: optional list of true changepoint times to
        overlay as vertical reference lines on every stage
    """
    n = len(stages)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, steady) in zip(axes, stages.items()):
        ax.plot(t, C, color="gray", alpha=0.4, linewidth=1)
        ax.scatter(t[steady], C[steady], s=10, color="orange", label="steady")
        ax.scatter(t[~steady], C[~steady], s=10, color="blue", label="transient")
        if true_change_times is not None:
            for xi in true_change_times:
                ax.axvline(xi, color="black", linestyle="--", alpha=0.4)
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    plt.show()