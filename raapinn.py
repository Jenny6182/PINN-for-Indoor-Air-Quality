import sys
import io
import csv
import contextlib
from datetime import datetime
from pathlib import Path
import refactored_single_simple_pinn as pinn

result = pinn.main(
                str("./varying_datasets/iaq_co2_varying_Q.csv"),
                plot_output_path=str("diagnostic.png"),
            )