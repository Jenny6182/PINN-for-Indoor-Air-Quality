

"""
1. combination of Q and S values
2. put into a list of tuples
3. call main function while iterating that list
4. save the useful results
"""

import sys
import numpy as np
import experiment.pipelines.raapinn as rp
import os
# Get the current working directory
current_dir = os.getcwd()
print(current_dir)

from itertools import product
from pathlib import Path
import io
from experiment.scripts.batch_train import Tee
import contextlib
import csv

datasets = ["./data/datasets/varying_pinn_datasets/varying_Q.csv", 
            "./data/datasets/varying_pinn_datasets/varying_S.csv"]
Q_values = [50, 100, 200, 500]
S_values = [1e4, 1e5, 1e6]
combinations = product(datasets, Q_values, S_values)

all_results = []

for path, Q_init, S_init in combinations:
    # name this run
    stem = Path(path).stem # get file name without extension
    run_dir = Path(f"results/sensitivity_analysis/{stem}_Q{Q_init}_S{S_init}")
    run_dir.mkdir(parents=True, exist_ok=True) # create if doesn't exist

    # set up buffer to write results
    log_buffer = io.StringIO() # a file stream
    tee = Tee(sys.__stdout__, log_buffer) # log_buffer is the stream2 that we will write into log file

    with contextlib.redirect_stdout(tee): # redirect stdout to tee class
        print(f"Running: Q_init={Q_init}, S_init={S_init}, path={path}")
        result = rp.main( # call raapinn main function
            path=path,
            log_Q_init=np.log(Q_init),
            log_S_init=np.log(S_init),
            run_dir=run_dir,
        )

    # save log
    with open(run_dir / "run.log", "w") as f:
        f.write(log_buffer.getvalue())

    # save summary
    with open(run_dir / "summary.csv", "w", newline="") as f: # open a summary.csv to record all results
        # add config info to each row so you know what produced this result
        rows = []
        if not result:
                print(f"No changepoints detected for Q_init={Q_init}, S_init={S_init}")
        else:
            for r in result:
                row = {
                    "path": path,
                    "Q_init": Q_init,
                    "S_init": S_init,
                    "tau": r["tau"],
                    "Q_minus": r["Q_minus"],
                    "Q_plus": r["Q_plus"],
                    "S_minus": r["S_minus"],
                    "S_plus": r["S_plus"],
                }
                rows.append(row)
                all_results.extend(rows)
            
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open("results/sensitivity_analysis/master_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)