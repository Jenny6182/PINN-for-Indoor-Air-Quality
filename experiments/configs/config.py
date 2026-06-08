# known physical constants 
V = 100.0   # zone volume, m^3
C_out = 420.0   # outdoor CO2, ppm
C0 = 500.0   # initial CO2, ppm  (first data point)

# pick a seed and always use it to make the initial weights, for reproducibility while tuning hyperparams
SEED = 42