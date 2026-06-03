import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import the data into a pandas dataframe
df = pd.read_csv("iaq_co2_simple.csv") # read the csv

print("DONE reading")
print("A few data: ", df.head())
print("Column names: ", df.columns)
print("Shape of the data: ", df.shape)

# Extracting the t hours column and using .values or .to_numpy to change df into numpy array
# time[i] and c_meas[i] will correspond to same data point
t_np = df["t_hours"].values # input, time, a np array
c_np = df["C_meas_ppm"].values # target, c_meas_ppm, a np array

# Reshape the numpy arrays so nn's expectation shape match actual shape (N,1)
t_np = t_np.reshape(-1, 1).astype(np.float32) # -1 is unknown dim, calculate needed rows based on number of element, 1 is 1 column
c_np = c_np.reshape(-1, 1).astype(np.float32)

# make training and test sets, t is input, c is output
# split training and testing cuz small dataset so no validation set
t_train_np, t_test_np, c_train_np, c_test_np = train_test_split(
    t_np, c_np, test_size=0.2, random_state=42
)

def read_data_nparray(path, input, output):
    """Given a path to csv file, read it and return tuple of input and output numpy arrays"""
    # Import the data into a pandas dataframe
    df = pd.read_csv(path) # read the csv

    # Inform progress
    print("DONE reading")
    print("A few data: ", df.head())
    print("Column names: ", df.columns)
    print("Shape of the data: ", df.shape)

    # Get the input and output rows
    x_np = df[input].values # get input into a np array
    y_np = df[output].values # get target into a np array

    # Reshape the numpy arrays and turn them into float type
    x_np = x_np.reshape(-1, 1).astype(np.float32) # -1 is unknown dim, calculate needed rows based on number of element, 1 is 1 column
    y_np = y_np.reshape(-1, 1).astype(np.float32)

    return x_np, y_np

def train_test_split(x_np, y_np):
    """Given x and y numpy arrays, return training and testing split in tuple of 4
       returns 0: x train, 1: x test, 2: y train, 3: y test
    """
    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
        x_np, y_np, test_size=0.2, random_state=42
    )
    return x_train_np, x_test_np, y_train_np, y_test_np

t_min  = t_train_np.min() 
t_max  = t_train_np.max()

c_mean = c_train_np.mean()
c_std  = c_train_np.std()


# --- normalization ---
"""
(should only be getting min max std mean of x and y 
only from training data to avoid data leak)
"""
# Normalize formulas:
# x-x_min / x_max-x_min or x-x_min / x_std bc trying to make x_std = 1

def norm_max_min(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min + 1e-8)
    # adding 1e-8 a very small number to prevent division by zero

def norm_mean_std(x, x_mean, x_std):
    return (x - x_mean) / x_std

def denorm(x_hat, x_mean, x_std):
    return x_hat * x_std + x_mean # To reverse normalization, multiply by std and add mean


# --- normalization ---
"""
(getting min max std mean of time and co2 
only from training data to avoid data leak)
"""
# usually nn exxpect input from 0 to 1 to be stable (cuz gradient could be very big) so we'll normalize time too
t_min  = t_train_np.min() 
t_max  = t_train_np.max()

c_mean = c_train_np.mean()
c_std  = c_train_np.std()


# normalize training tensors
T_train = torch.tensor(norm_t(t_train_np), dtype=torch.float32)
C_train = torch.tensor(norm_c(c_train_np), dtype=torch.float32)

# ------ make collocation points that should spread uniformly over the training time window ------
t_col_np = np.linspace(t_min, t_max, N_COLLOC).reshape(-1, 1).astype(np.float32) #make np list
T_col = torch.tensor(norm_t(t_col_np), dtype=torch.float32, requires_grad=True) # turn np list into tensor, requires gradient=True to compute dC/dt



# Turn np arrays into tensors
t_data = torch.tensor(t_np, dtype=torch.float32)
c_data = torch.tensor(c_np, dtype=torch.float32)


# Save the tensors
# torch.save(t_data, "X.pt") # .pt is file for any pytorch project (this is individually)
# torch.save(c_data, "Y.pt")
torch.save({"X": t_data, "Y": c_data}, "data.pt") # saving together





