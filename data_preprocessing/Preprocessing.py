"""
Preprocessing.py includes all utilities handling normalizing data, loading, and saving data:

- normalize(): take in numpy array, return the normalized numpy array that is clipped into [0, 1]
- standardize(): take in numpy array, return the standardized numpy array that is adjusted to have
                 mean of 0 and standard deviation of 1
- compute_stats(): compute all commonly used stats of numpy arrays x and y
- load_csv(): Given the path to a csv, return the indicated x_col and y_col from that dataset as numpy arrays
- split_data(): Given the test percentage and x, y numpy array and a random seed,
                return X_train, X_test, y_train, y_test
- to_tensors(): Given x, y numpy arrays, return X, Y tensors
- save_dataset(): Save given tensors X, Y at given path
- load_dataset(): Load dataset with X, Y tensors from path
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_split

def normalize(x):
    """Given a numpy array, return the normalized numpy array that is clipped into [0, 1].
       Also known as min-max normalization. """
    # +1e-8 in denominator to avoid division by 0
    # formula: x-x_min / x_max-x_min
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def standardize(x):
    """Given a numpy array, return the standardized numpy array that is adjusted to have
       mean of 0 and standard deviation of 1. """
    # formula: x-x_min / x_std
    return (x - x.mean()) / x.std()

def normalize_with_stats(x, x_min, x_max):
    """Same as normalize, but used specifically for test set
    The stats applied are from training stats, don't use test set's stats."""
    return (x - x_min) / (x_max - x_min + 1e-8)

def standardize_with_stats(x, mean, std):
    """Same as normalize, but used specifically for test set
    The stats applied are from training stats, don't use test set's stats."""
    return (x - mean) / (std + 1e-8)

def compute_stats(x_train, y_train):
    """Given numpy arrays, return min, max, mean, std of that array"""
    return {
        "t_min": x_train.min(),
        "t_max": x_train.max(),
        "y_mean": y_train.mean(),
        "y_std": y_train.std()
    }

# .values() give numpy array, .to_numpy() also give numpy array
def load_csv(path, x_col, y_col):
    """Given a path to csv file, read it and return tuple of x, y numpy arrays"""
    df = pd.read_csv(path) # read the csv

    x = df[x_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = df[y_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    # -1 is unknown dim, calculate needed rows based on number of element, 1 is 1 column
    # turn into numpy array with floats, reshape to (481, ) instead of (481, 1) shape

    return x, y

def load_columns(path, cols):
     """Given a path to csv file, read col and return col as an numpy array"""
     extras = {}
     if cols:
         df = pd.read_csv(path)
         for col in cols:
             extras[col] = df[col].to_numpy(dtype=np.float32)
     return extras

def split_data(x, y, test_size=0.2, seed=42):
    """Given the test percentage and x, y numpy array and a random seed,
    return X_train, X_test, y_train, y_test"""
    return sk_split(x, y, test_size=test_size, random_state=seed)

def to_tensors(x, y):
    """Given x, y numpy arrays, return X, Y tensors"""
    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)

    return X, Y

def save_dataset(path, X, Y):
    """Save given tensors at path"""
    torch.save({"X": X, "Y": Y}, path) # saving together, .pt is file for any pytorch project (this is individually)

def load_dataset(path):
    """Load dataset with X, Y tensors from path"""
    return torch.load(path)

def prepare_training_data(path, x_col, y_col, extra_cols=None, test_size=0.2, seed=42):
    """
    Full preprocessing pipeline. Returns everything needed to start training.
    extra_cols: list of additional column names to load (e.g. ["Q_true", "S_true"])
    """
    # load raw data
    x, y = load_csv(path, x_col, y_col)

    # load extra columns if requested (e.g. ground truth Q and S for piecewise case)
    extras = load_columns(extra_cols)

    # split dataset bc stats must be computed from training set only
    x_train, x_test, y_train, y_test = split_data(x, y, test_size, seed)

    # compute stats from training data only
    stats = compute_stats(x_train, y_train)

    # normalize using training stats
    x_train_norm = normalize_with_stats(x_train, stats["t_min"], stats["t_max"])
    y_train_norm = standardize_with_stats(y_train, stats["y_mean"], stats["y_std"])

    # convert to tensors
    T_train, C_train = to_tensors(x_train_norm, y_train_norm)

    return {
        "t_np": x, # full raw time array, for plotting
        "c_np": y, # full raw C array, for plotting
        "t_train_np": x_train, # raw training time, for collocation point creation
        "c_train_np": y_train, # raw training C
        "t_test_np": x_test,
        "c_test_np": y_test,
        "T_train": T_train, # normalised tensors fed to training loop
        "C_train": C_train,
        "stats": stats, # t_min, t_max, c_mean, c_std
        **extras, # Q_true_np, S_true_np if requested
    }
