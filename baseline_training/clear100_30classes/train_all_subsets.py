#!/usr/bin/env python
"""
Train model on all CLEAR-100 temporal bins (30 classes).

Trains sequentially on 5 two-year bins (year_1-2 through year_9-10) and saves 
best checkpoints for each.

Usage:
    python train_all_subsets.py
    python train_all_subsets.py --max_epochs 15 --batch_size 64
    python train_all_subsets.py --devices 0
    python train_all_subsets.py --years year_1-2 year_3-4

Output:
    Checkpoints: model_checkpoints/{model}_{bin}-best-acc-epoch-XX.ckpt
    Logs: logs/tensorboard/ and logs/csv/
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time
import random


# CLEAR-100 merged year bins
ALL_YEARS_MERGED = ['year_1-2', 'year_3-4', 'year_5-6', 'year_7-8', 'year_9-10']

# Individual years (for compatibility)
ALL_YEARS_INDIVIDUAL = [f'year_{i}' for i in range(1, 11)]


def run_training(year_name, args):
    """
    Run training for a single year.
    
    Returns:
        bool: True if training succeeded
    """
    print("=" * 80)
    print(f"Training year: {year_name}")
    print("=" * 80)
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "train_clear100.py",
        "--year_name", year_name,
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ]
    
    # Add optional arguments
    if args.data_root:
        cmd.extend(["--data_root", args.data_root])
    if args.model_name:
        cmd.extend(["--model_name", args.model_name])
    if args.checkpoint_dir:
        cmd.extend(["--checkpoint_dir", args.checkpoint_dir])
    if args.log_dir:
        cmd.extend(["--log_dir", args.log_dir])
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    if args.no_augmentation:
        cmd.append("--no_augmentation")
    if args.early_stopping:
        cmd.append("--early_stopping")
    if args.early_stopping_patience:
        cmd.extend(["--early_stopping_patience", str(args.early_stopping_patience)])
    if args.accelerator:
        cmd.extend(["--accelerator", args.accelerator])
    if args.devices:
        cmd.extend(["--devices", args.devices])
    if args.precision:
        cmd.extend(["--precision", args.precision])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.deterministic:
        cmd.append("--deterministic")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print()
        print(f"[SUCCESS] Training completed: {year_name}")
        print(f"Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print()
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print()
        print(f"[ERROR] Training failed: {year_name}: {e}")
        print(f"Failed after {duration/60:.1f} minutes")
        print()
        return False
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user")
        print()
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model on all CLEAR-100 temporal bins (30 classes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Year selection
    parser.add_argument(
        "--years",
        nargs="+",
        default=ALL_YEARS_MERGED,
        help="Years/bins to train on"
    )
    
    # Option to use individual years
    parser.add_argument(
        "--use-individual-years",
        action="store_true",
        help="Use individual years (year_1 through year_10) instead of merged bins"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../datasets/CLEAR100_30classes",
        help="Root directory of CLEAR-100 dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="apple/mobilevit-xx-small",
        help="Model architecture"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=15, help="Maximum epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=7, help="Early stopping patience")
    
    # Hardware arguments
    parser.add_argument("--accelerator", type=str, help="Accelerator type (auto/gpu/cpu)")
    parser.add_argument(
        "--devices",
        type=str,
        help="GPU device(s) to use. Examples: '1' (use 1 GPU), '0' (use GPU 0)"
    )
    parser.add_argument("--precision", type=str, help="Training precision (32/16-mixed/bf16-mixed)")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_checkpoints", help="Checkpoint directory")
    
    # Reproducibility - use random seed by default
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if not specified)")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate random seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 999999)
        print(f"Using random seed: {args.seed}")
    
    # Override years if using individual years
    if args.use_individual_years:
        args.years = ALL_YEARS_INDIVIDUAL
    
    print("CLEAR-100: TRAIN ALL TEMPORAL BINS/YEARS (30 classes)")
    print(f"Dataset: CLEAR-100")
    print(f"Data root: {args.data_root}")
    print(f"Model: {args.model_name}")
    print(f"Seed: {args.seed}")
    if args.use_individual_years:
        print(f"Mode: Individual years (year_1 through year_10)")
    else:
        print(f"Mode: Merged 2-year bins")
    print(f"Years/bins to train: {len(args.years)}")
    for year in args.years:
        print(f"  - {year}")
    
    # Verify data directory exists
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {args.data_root}")
        print("Please run preprocessing first or specify correct --data_root")
        return 1
    
    # Train each year
    results = {}
    total_start = time.time()
    
    for i, year in enumerate(args.years, 1):
        print(f"YEAR {i}/{len(args.years)}: {year.upper()}")
        
        success = run_training(year, args)
        results[year] = success
        
        if not success:
            print(f"[WARNING] Year {year} failed but continuing...")
    
    # Print summary
    total_elapsed = time.time() - total_start
    
    print("TRAINING SUMMARY")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"\nResults:")
    
    succeeded = sum(1 for s in results.values() if s)
    failed = len(results) - succeeded
    
    for year, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"{status} {year}")
    
    print(f"\nSummary: {succeeded}/{len(results)} succeeded")
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
