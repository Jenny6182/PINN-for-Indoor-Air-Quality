"""
test_one.py
-----------
Runs the PINN on a single CSV to verify everything works before
committing to a full batch run.

Saves output to results/<stem>/ just like batch_train would,
so you can check the log, plot, and summary are all correct.

Change TEST_FILE to whichever file you want to try.
"""

import csv
import io
from datetime import datetime
from pathlib import Path

import refactored_single_simple_pinn as pinn
from batch_train import Tee

DATA_FOLDER    = Path("./datasets/")
RESULTS_FOLDER = Path("./results/")

# grabs the first file alphabetically — change this to test a specific one
# e.g. TEST_FILE = DATA_FOLDER / "iaq_Q800_S0.05.csv"
TEST_FILE = sorted(DATA_FOLDER.glob("*.csv"))[0]

stem    = TEST_FILE.stem
run_dir = RESULTS_FOLDER / stem
run_dir.mkdir(parents=True, exist_ok=True)

print(f"Test run: {TEST_FILE.name}")
print(f"Output -> {run_dir.resolve()}\n")

import sys
import contextlib

log_buffer = io.StringIO()
tee        = Tee(sys.__stdout__, log_buffer)

with contextlib.redirect_stdout(tee):
    print(f"{'='*60}")
    print(f"  [test]  {TEST_FILE.name}")
    print(f"{'='*60}")
    print()

    result = pinn.main(
        str(TEST_FILE),
        plot_output_path=str(run_dir / "diagnostic.png"),
    )

# write log
with open(run_dir / "run.log", "w", encoding="utf-8") as f:
    f.write(f"Run started : {datetime.now().isoformat()}\n")
    f.write(f"Input file  : {TEST_FILE.resolve()}\n")
    f.write("\n")
    f.write(log_buffer.getvalue())

# write summary
if result:
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

print(f"\n--- Result dict ---")
for k, v in result.items():
    print(f"  {k}: {v}")

print(f"\nFiles saved to: {run_dir.resolve()}")