"""
Training script for CLEAR-10 multi-class classification (10 classes).

Usage:
    python train_clear10.py --year_name year_1-2
    python train_clear10.py --year_name year_5 --max_epochs 15 --batch_size 64

Output:
    Checkpoints: model_checkpoints/{model}_{year}-best-acc-epoch-XX.ckpt
    Logs: logs/tensorboard/ and logs/csv/
"""

import os
import sys
import argparse
import yaml
import psutil
from pathlib import Path
from typing import Optional
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    RichModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from data_module_clear10 import CLEAR10DataModule, auto_detect_num_workers
from model_clear10 import CLEAR10Classifier
import warnings


def parse_devices(devices_str: str):
    """
    Parse devices string for PyTorch Lightning.
    Examples: "1" -> 1, "0" -> [0], "0,1" -> [0,1]
    """
    if not devices_str:
        return 1
    
    if ',' in devices_str:
        try:
            gpu_ids = [int(x.strip()) for x in devices_str.split(',')]
            return gpu_ids
        except ValueError:
            warnings.warn(f"Invalid devices format: {devices_str}, using default")
            return 1
    else:
        try:
            single_device = int(devices_str.strip())
            if single_device == 1:
                return 1
            else:
                return [single_device]
        except ValueError:
            warnings.warn(f"Invalid devices format: {devices_str}, using default")
            return 1


def setup_callbacks(args, file_base: str):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{file_base}-best-acc-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if args.early_stopping:
        early_stop = EarlyStopping(
            monitor='val_acc',
            patience=args.early_stopping_patience,
            mode='max',
            verbose=True,
        )
        callbacks.append(early_stop)
    
    # Progress bar
    callbacks.append(RichProgressBar())
    callbacks.append(RichModelSummary(max_depth=2))
    
    return callbacks, checkpoint_callback


def setup_loggers(args, file_base: str):
    """Setup loggers."""
    loggers = []
    
    # TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="tensorboard",
        version=file_base,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # CSV logger
    csv_logger = CSVLogger(
        save_dir=args.log_dir,
        name="csv",
        version=file_base,
    )
    loggers.append(csv_logger)
    
    return loggers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CLEAR-10 classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="../../datasets/CLEAR10")
    parser.add_argument("--year_name", type=str, required=True, help="Year to train on")
    parser.add_argument("--num_workers", type=int, default=-1, help="Data loading workers (-1=auto)")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable augmentation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="mobilevit-xxs")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--freeze_backbone", action="store_true")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    
    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--precision", type=str, default="16-mixed")
    
    # Callbacks
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    
    # Reproducibility - use random seed by default
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if not set)")
    parser.add_argument("--deterministic", action="store_true")
    
    # Resume and test
    parser.add_argument("--resume_from_checkpoint", type=str, help="Checkpoint to resume from")
    parser.add_argument("--run_test", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Parse devices
    args.devices = parse_devices(args.devices)
    
    # Auto-detect num_workers
    if args.num_workers == -1:
        args.num_workers = auto_detect_num_workers(verbose=True)
    
    # Convert augmentation flag
    args.augmentation = not args.no_augmentation
    
    return args


def main():
    """Main training function."""
    args = parse_args()
    
    # Generate random seed if not provided (experiments use random seeds)
    if args.seed is None:
        args.seed = random.randint(0, 999999)
        print(f"Using random seed: {args.seed}")
    
    # Set seeds
    pl.seed_everything(args.seed, workers=True)
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    print("CLEAR-10 TRAINING")
    print(f"Year: {args.year_name}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print(f"Data augmentation: {args.augmentation}")
    
    # Load data
    print("Loading data...")
    data_module = CLEAR10DataModule(
        data_root=args.data_root,
        year_name=args.year_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
    )
    
    data_module.prepare_data()
    data_module.setup()
    
    stats = data_module.get_dataset_stats()
    print(f"[SUCCESS] Dataset loaded")
    print(f"Year: {stats['year_name']}")
    print(f"Classes: {stats['num_classes']}")
    print(f"Train: {stats['train_size']}")
    print(f"Val: {stats['val_size']}")
    print(f"Test: {stats['test_size']}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = CLEAR10Classifier(
        model_name=args.model_name,
        num_classes=10,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout_rate,
    )
    
    # File naming
    year_name = args.year_name
    model_prefix = args.model_name.replace("/", "-").replace("apple-", "").replace("microsoft-", "")
    file_base = f"{model_prefix}_{year_name}"
    
    print(f"\nFile prefix: {file_base}")
    print(f"Checkpoints: {file_base}-best-acc-epoch-XX.ckpt")
    print(f"Logs: logs/{{tensorboard,csv}}/{file_base}/")
    
    # Setup callbacks and loggers
    print(f"\nSetting up callbacks...")
    callbacks, checkpoint_callback = setup_callbacks(args, file_base)
    
    print(f"Setting up loggers...")
    loggers = setup_loggers(args, file_base)
    
    # Initialize trainer
    print(f"Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=args.deterministic,
        benchmark=not args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        enable_model_summary=False,
    )
    
    print("STARTING TRAINING")
    
    try:
        trainer.fit(model, data_module)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        raise
    
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print("TRAINING COMPLETE")
    print(f"\nDuration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Epochs: {trainer.current_epoch + 1}")
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nBest model saved: {best_model_path}")
    
    # Run test evaluation
    if args.run_test:
        print("TEST EVALUATION")
        
        test_results = trainer.test(model, data_module, ckpt_path=best_model_path, verbose=False)
        
        if test_results:
            test_metrics = test_results[0]
            
            # Save results
            results_dir = Path(args.checkpoint_dir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            exp_name = args.experiment_name or args.year_name
            results_file = results_dir / f"{exp_name}_test_results.json"
            
            results_data = {
                'year_name': args.year_name,
                'experiment_name': exp_name,
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'best_checkpoint': str(best_model_path),
                'training_duration_seconds': duration,
                'seed': args.seed,
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\nTest results saved: {results_file}")
            print(f"Test accuracy: {test_metrics.get('test_acc', 0):.4f}")
    
    print("DONE")


if __name__ == "__main__":
    main()
