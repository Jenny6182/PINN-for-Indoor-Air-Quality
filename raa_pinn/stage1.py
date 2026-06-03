import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d
from varying_pinn.varying_pinn import load_data
import matplotlib.pyplot as plt

def stage1_scan(t, C_meas, V, C_out, window_size=10):
    # smooth measurements first to reduce noise in derivative
    c_np = C_meas.flatten()
    t_np = t.flatten()

    C_smooth = gaussian_filter1d(c_np, sigma=1)

    dC_dt    = np.gradient(C_smooth, t_np)   # numerical derivative

    n       = len(t)
    scores  = np.full(n, np.nan)
    half    = window_size // 2

    for i in range(half, n - half):
        idx = slice(i - half, i + half)

        # ODE: V*dC/dt = Q*(C_out - C) + S
        # rewrite as linear system: [C_out - C, 1] @ [Q, S] = V*dC/dt
        A = np.column_stack([C_out - C_smooth[idx], np.ones(window_size)])
        b = V * dC_dt[idx]

        # least squares fit of Q and S on this window
        params, residuals, _, _ = lstsq(A, b, rcond=None)

        print("calculating {i}th score")
        # residual is the physics inconsistency signal
        scores[i] = np.mean((A @ params - b) ** 2)

    return scores

def norm_scores(scores):
    # Global Min-Max Normalization
    scores_min = scores.min()
    print(min)
    scores_max = scores.max()
    return (scores - scores_min) / (scores_max-scores_min)



def plot_scores(t_np, scores):
    print(scores.shape, t_np.shape)
    plt.plot(t_np, scores)
    plt.show()

t_np, c_np, t_train_np, c_train_np, t_test_np, c_test_np, Q_true_np, S_true_np = load_data("./varying_datasets/iaq_co2_varying_Q.csv")
V     = 100.0   # zone volume   [m^3]
C_out = 420.0   # outdoor CO2   [ppm]
scores = stage1_scan(t_np, c_np, V, C_out)

clean_scores = np.array(np.nan_to_num(scores))

print(clean_scores)

plot_scores(t_np.flatten(), clean_scores)
plot_scores(t_np.flatten(), norm_scores(clean_scores))