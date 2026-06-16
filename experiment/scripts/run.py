"""
Unified script to run any PINN pipeline (simple, varying, raapinn, one_raapinn).

Usage:
    # Single run with defaults:
    python run.py --model-type raapinn --data-path ./data/datasets/varying_pinn_datasets/iaq_co2_varying_Q.csv
    
    # With custom config:
    python run.py --model-type raapinn --config experiment/configs/raapinn_config.yaml --data-path data.csv
    
    # Batch sensitivity analysis:
    python run.py --batch experiment/configs/batch_sweep.yaml
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
from itertools import product

from experiment.pipelines import get_pipeline
from experiment.scripts.batch_train import Tee
import sys
import io
import contextlib


def run_single(model_type, data_path, config_path=None, run_dir=None, **kwargs):
    """Run a single pipeline execution."""
    print(f"\n{'='*60}")
    print(f"Running {model_type} PINN")
    print(f"Data: {data_path}")
    print(f"{'='*60}")
    
    # Get pipeline
    pipeline = get_pipeline(model_type, config_path=config_path, run_dir=run_dir)
    pipeline.save_config()
    
    # Set up logging
    log_buffer = io.StringIO()
    tee = Tee(sys.__stdout__, log_buffer)
    
    # Run with output captured
    with contextlib.redirect_stdout(tee):
        results = pipeline.run(data_path, **kwargs)
    
    # Save log
    with open(pipeline.run_dir / "run.log", "w") as f:
        f.write(log_buffer.getvalue())
    
    print(f"\nResults saved to: {pipeline.run_dir}")
    return results


def run_batch(batch_config_path):
    """Run batch experiment from config file."""
    with open(batch_config_path) as f:
        batch_config = yaml.safe_load(f)
    
    model_type = batch_config["model_type"]
    data_paths = batch_config["data_paths"]
    parameter_space = batch_config.get("parameters", {})
    
    print(f"\n{'='*60}")
    print(f"Batch Run: {model_type} PINN")
    print(f"Datasets: {len(data_paths)}")
    print(f"Parameter combinations: {np.prod([len(v) for v in parameter_space.values()])}")
    print(f"{'='*60}")
    
    all_results = []
    
    # Generate all parameter combinations
    if parameter_space:
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        param_combinations = product(*param_values)
    else:
        param_combinations = [{}]
    
    # Run each combination with each dataset
    for data_path in data_paths:
        for params in param_combinations:
            param_dict = dict(zip(param_names, params)) if param_names else {}
            run_name = f"{Path(data_path).stem}_" + "_".join(
                f"{k}={v}" for k, v in param_dict.items()
            )
            run_dir = Path(batch_config.get("output_dir", "results")) / run_name
            
            try:
                results = run_single(
                    model_type,
                    data_path,
                    config_path=batch_config.get("config_path"),
                    run_dir=run_dir,
                    **param_dict
                )
                all_results.append({
                    'data_path': data_path,
                    'params': param_dict,
                    'success': True,
                    'results': results
                })
            except Exception as e:
                print(f"ERROR: {run_name}: {e}")
                all_results.append({
                    'data_path': data_path,
                    'params': param_dict,
                    'success': False,
                    'error': str(e)
                })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Unified PINN pipeline runner"
    )
    
    # Single run options
    parser.add_argument(
        "--model-type",
        choices=['simple', 'varying', 'raapinn', 'one_raapinn'],
        help="Model type to run"
    )
    parser.add_argument(
        "--data-path",
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--config",
        help="Path to model config YAML file"
    )
    parser.add_argument(
        "--run-dir",
        help="Output directory (auto-generated if not provided)"
    )
    
    # Batch run options
    parser.add_argument(
        "--batch",
        help="Path to batch config YAML file for sensitivity analysis"
    )
    
    # Additional model parameters
    parser.add_argument(
        "--log-Q-init",
        type=float,
        help="Initial log(Q) for RAA-PINN models"
    )
    parser.add_argument(
        "--log-S-init",
        type=float,
        help="Initial log(S) for RAA-PINN models"
    )
    
    args = parser.parse_args()
    
    # Batch run
    if args.batch:
        results = run_batch(args.batch)
        print(f"\nBatch complete. {sum(r['success'] for r in results)}/{len(results)} successful.")
    
    # Single run
    elif args.model_type and args.data_path:
        kwargs = {}
        if args.log_Q_init is not None:
            kwargs['log_Q_init'] = args.log_Q_init
        if args.log_S_init is not None:
            kwargs['log_S_init'] = args.log_S_init
        
        run_single(
            args.model_type,
            args.data_path,
            config_path=args.config,
            run_dir=Path(args.run_dir) if args.run_dir else None,
            **kwargs
        )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  Single run:")
        print("    python run.py --model-type raapinn --data-path data.csv")
        print("  Batch run:")
        print("    python run.py --batch batch_config.yaml")


if __name__ == "__main__":
    main()
