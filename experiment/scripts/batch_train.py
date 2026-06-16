"""
batch_train.py
--------------
Runs the PINN on every .csv in a folder.
For each file, creates its own subfolder under results/ containing:
    run.log        - full console output including the header
    diagnostic.png - the PINN diagnostic plot
    summary.csv    - key numbers for that run (true vs estimated Q, S, losses)

results/
    iaq_Q800_S0.05/
        run.log
        diagnostic.png
        summary.csv
    iaq_Q400_S0.10/
        run.log
        diagnostic.png
        summary.csv
    ...

Usage:
    python batch_train.py                        # data in ./datasets/, results in ./results/
    python batch_train.py ./my_data/ ./my_out/   # custom folders
"""

import sys
import io
import csv
import contextlib
from datetime import datetime
from pathlib import Path



class Tee:
    """Terminal and log buffer together in one, pass in terminal and log buffer as stream1 and stream2"""
    def __init__(self, stream1, stream2):
        self.s1 = stream1
        self.s2 = stream2

    def write(self, data):  # writing to both streams
        self.s1.write(data)
        self.s2.write(data)

    def flush(self): # flushing both streams
        self.s1.flush()
        self.s2.flush()


# def run_folder(data_folder="./datasets/", results_folder="./results/"):
#     data_folder = Path(data_folder)
#     results_folder = Path(results_folder)
#     results_folder.mkdir(parents=True, exist_ok=True)

#     csv_files = sorted(data_folder.glob("*.csv"))
#     if not csv_files:
#         print(f"No .csv files found in {data_folder.resolve()}")
#         return

#     print(f"Found {len(csv_files)} CSV file(s) in {data_folder.resolve()}")
#     print(f"Results will be saved to: {results_folder.resolve()}\n")

#     for i, csv_path in enumerate(csv_files, 1):
#         stem = csv_path.stem
#         run_dir = results_folder / stem
#         run_dir.mkdir(parents=True, exist_ok=True)

#         log_buffer = io.StringIO()
#         tee = Tee(sys.__stdout__, log_buffer)  # prints to terminal AND buffer

#         with contextlib.redirect_stdout(tee):
#             print(f"{'='*60}")
#             print(f"  [{i}/{len(csv_files)}]  {csv_path.name}")
#             print(f"{'='*60}")
#             print()

#             result = pinn.main(
#                 str(csv_path),
#                 plot_output_path=str(run_dir / "diagnostic.png"),
#             )

#         # write log (everything that was also printed to terminal)
#         with open(run_dir / "run.log", "w", encoding="utf-8") as f:
#             f.write(f"Run started : {datetime.now().isoformat()}\n")
#             f.write(f"Input file  : {csv_path.resolve()}\n")
#             f.write("\n")
#             f.write(log_buffer.getvalue())

#         # write per-run summary.csv
#         if result:
#             with open(run_dir / "summary.csv", "w", newline="") as f:
#                 writer = csv.DictWriter(f, fieldnames=list(result.keys()))
#                 writer.writeheader()
#                 writer.writerow(result)

#         print(f"\n  -> saved to {run_dir}\n")

#     print("All files processed.")


# if __name__ == "__main__":
#     data_folder = sys.argv[1] if len(sys.argv) > 1 else "./datasets/"
#     results_folder = sys.argv[2] if len(sys.argv) > 2 else "./results/"
#     run_folder(data_folder, results_folder)