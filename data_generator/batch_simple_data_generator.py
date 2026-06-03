import data_generator.iaq_co2_simple_simulator as iaq_co2_simple_simulator

from data_generator.iaq_co2_simple_simulator import run_iaq_simulation

Q_values = [100, 200, 400, 500, 700, 800]
S_values = [0.05, 0.1, 0.2, 0.3]

for Q in Q_values:
    for S in S_values:

        result = run_iaq_simulation(
            Q=Q,
            S_vol=S
        )

        print(result["C_ss"])