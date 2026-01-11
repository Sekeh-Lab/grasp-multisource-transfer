#!/usr/bin/env python
"""
train_all_subsets.py - Train model on all Yearbook decade subsets

This script trains the binary decade classification model sequentially on all subsets
of the Yearbook dataset and saves the best checkpoints for each.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time


# All Yearbook subsets
ALL_SUBSETS = [
    'before_1950s',
    '1950s_1960s',
    '1970s_1980s',
    '1990s_and_later'
]


def run_training(subset_name, args):
    """
    Run training for a single subset.
    
    Args:
        subset_name: Name of the subset to train on (e.g., "before_1950s")
        args: Command-line arguments
    
    Returns:
        bool: True if training succeeded, False otherwise
    """
    print("=" * 80)
    print(f"Training subset: {subset_name}")
    print("=" * 80)
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "train_yearbook.py",
        "--subset_name", subset_name,
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
    if args.seed != 42:
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
        print(f"Successfully completed training for {subset_name}")
        print(f"  Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print()
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print()
        print(f"Error training {subset_name}: {e}")
        print(f"  Failed after {duration/60:.1f} minutes")
        print()
        return False
    except KeyboardInterrupt:
        print()
        print("âš ï¸  Training interrupted by user!")
        print()
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model on all Yearbook decade subsets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subset selection
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=ALL_SUBSETS,
        choices=ALL_SUBSETS,
        help="Subsets to train on (default: all)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../datasets/Yearbook_Decades",
        help="Root directory of Yearbook dataset"
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
        help="GPU device(s) to use. Examples: '1' (use 1 GPU), '0' (use GPU 0), '0,1' (use GPUs 0 and 1)"
    )
    parser.add_argument("--precision", type=str, help="Training precision (32/16-mixed/bf16-mixed)")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_checkpoints", help="Checkpoint directory")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("YEARBOOK: TRAIN ALL DECADE SUBSETS")
    print("=" * 80)
    print(f"Dataset: Yearbook (Century of Portraits)")
    print(f"Data root: {args.data_root}")
    print(f"Model: {args.model_name}")
    print(f"Subsets to train: {len(args.subsets)}")
    for subset in args.subsets:
        print(f"  - {subset.replace('_', ' ').replace('s', 's ')}")
    print("=" * 80 + "\n")
    
    # Verify data directory exists
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data_root}")
        print("Please run preprocessing first or specify correct --data_root")
        return 1
    
    # Train each subset
    results = {}
    total_start = time.time()
    
    for i, subset in enumerate(args.subsets, 1):
        print(f"\n{'='*80}")
        print(f"SUBSET {i}/{len(args.subsets)}: {subset.upper().replace('_', ' ')}")
        print(f"{'='*80}\n")
        
        success = run_training(subset, args)
        results[subset] = success
        
        if not success:
            print(f"Subset {subset} failed but continuing with remaining subsets...")
    
    # Print summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"\nResults:")
    
    succeeded = sum(1 for s in results.values() if s)
    failed = len(results) - succeeded
    
    for subset, success in results.items():
        status = "Done" if success else "Fail"
        print(f"  {status} {subset.replace('_', ' ').replace('s', 's ')}")
    
    print(f"\nSummary: {succeeded}/{len(results)} succeeded")
    print("=" * 80 + "\n")
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        sys.exit(1)