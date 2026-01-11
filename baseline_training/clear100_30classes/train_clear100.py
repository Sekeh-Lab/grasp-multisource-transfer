"""
train_clear100.py - Training script for CLEAR-100 multi-class classification

This script trains a model on a single year of the CLEAR-100 dataset (11-class classification).
Use train_all_subsets.py to train on all 10 years sequentially.
"""

import os
import sys
import argparse
import yaml
import psutil
from pathlib import Path
from typing import Optional
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import re
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    RichModelSummary,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from data_module_clear100 import CLEAR100DataModule
from model_clear100 import CLEAR100Classifier
import warnings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_devices(devices_str: str):
    """
    Parse devices string into appropriate format for PyTorch Lightning.
    """
    if not devices_str:
        return 1
    
    # Check if it contains commas (multiple GPUs)
    if ',' in devices_str:
        # Parse as list of GPU IDs
        try:
            gpu_ids = [int(x.strip()) for x in devices_str.split(',')]
            return gpu_ids
        except ValueError:
            print(f"Warning: Could not parse devices '{devices_str}', using default (1 GPU)")
            return 1
    else:
        # Single value
        try:
            device_val = int(devices_str)
            # If 0, use GPU 0; if >1, use that many GPUs
            if device_val == 0:
                return [0]  # Use GPU 0 specifically
            else:
                return device_val  # Use this many GPUs
        except ValueError:
            print(f"Warning: Could not parse devices '{devices_str}', using default (1 GPU)")
            return 1

class CustomLearningRateMonitor(Callback):
    """Custom learning rate monitor that avoids the torch.tensor() warning."""
    
    def __init__(self, logging_interval='step'):
        super().__init__()
        self.logging_interval = logging_interval
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.logging_interval == 'step':
            self._log_lr(trainer, pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        if self.logging_interval == 'epoch':
            self._log_lr(trainer, pl_module)
    
    def _log_lr(self, trainer, pl_module):
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                
                if isinstance(lr, torch.Tensor):
                    lr_value = float(lr.detach().cpu().item())
                else:
                    lr_value = float(lr)
                
                if len(optimizer.param_groups) == 1:
                    pl_module.log('lr', lr_value, on_step=True, on_epoch=False, 
                                 prog_bar=True, logger=True)
                else:
                    pl_module.log(f'lr-pg{i}', lr_value, on_step=True, on_epoch=False,
                                 prog_bar=(i == 0), logger=True)


class MetricsVisualizationCallback(Callback):
    """Callback to visualize training metrics every N epochs."""
    
    def __init__(self, plot_every_n_epochs=5, save_dir="./plots", file_base="model"):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_dir = Path(save_dir)
        self.file_base = file_base
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        if trainer.callback_metrics:
            train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
            val_loss = trainer.callback_metrics.get('val_loss', None)
            train_acc = trainer.callback_metrics.get('train_acc', None)
            val_acc = trainer.callback_metrics.get('val_acc', None)
            
            if all(x is not None for x in [train_loss, val_loss, train_acc, val_acc]):
                self.epochs.append(current_epoch)
                self.train_losses.append(float(train_loss.item() if torch.is_tensor(train_loss) else train_loss))
                self.val_losses.append(float(val_loss.item() if torch.is_tensor(val_loss) else val_loss))
                self.train_accs.append(float(train_acc.item() if torch.is_tensor(train_acc) else train_acc))
                self.val_accs.append(float(val_acc.item() if torch.is_tensor(val_acc) else val_acc))
        
        if (current_epoch + 1) % self.plot_every_n_epochs == 0 and len(self.epochs) > 0:
            self._plot_metrics(current_epoch)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _plot_metrics(self, current_epoch):
        try:
            if len(self.epochs) < 2:
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(self.epochs, self.train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
            ax1.plot(self.epochs, self.val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax1.set_title(f'Training and Validation Loss (Epoch {current_epoch + 1})', 
                         fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            train_accs_pct = [acc * 100 for acc in self.train_accs]
            val_accs_pct = [acc * 100 for acc in self.val_accs]
            ax2.plot(self.epochs, train_accs_pct, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
            ax2.plot(self.epochs, val_accs_pct, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Training and Validation Accuracy (Epoch {current_epoch + 1})', 
                         fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 100])
            
            plt.tight_layout()
            
            plot_path = self.save_dir / f"{self.file_base}_metrics_epoch_{current_epoch + 1:03d}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            print(f"\nMetrics plot saved to: {plot_path}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
        finally:
            plt.close('all')


def get_optimal_num_workers() -> int:
    """
    Automatically determine optimal number of data loading workers based on system resources.
    
    Returns:
        int: Recommended number of workers (0 if insufficient resources detected)
    """
    try:
        # Get available memory and shared memory
        virtual_mem = psutil.virtual_memory()
        available_memory_gb = virtual_mem.available / (1024 ** 3)
        total_memory_gb = virtual_mem.total / (1024 ** 3)
        
        # Try to check shared memory (Linux only)
        shared_mem_gb = None
        if sys.platform.startswith('linux'):
            try:
                shm_stats = os.statvfs('/dev/shm')
                shared_mem_gb = (shm_stats.f_bavail * shm_stats.f_frsize) / (1024 ** 3)
            except:
                pass
        
        # Get CPU count
        cpu_count = os.cpu_count() or 1
        
        print(f"System resources detected:")
        print(f"   - CPUs: {cpu_count}")
        print(f"   - Total RAM: {total_memory_gb:.1f} GB")
        print(f"   - Available RAM: {available_memory_gb:.1f} GB")
        if shared_mem_gb is not None:
            print(f"   - Shared memory (/dev/shm): {shared_mem_gb:.2f} GB")
        
        # Conservative decision logic for limited memory environments
        if shared_mem_gb is not None and shared_mem_gb < 1.0:
            print(f"Insufficient shared memory ({shared_mem_gb * 1024:.0f} MB < 1 GB).")
            print(f"   Using num_workers=0 to avoid multiprocessing issues.")
            return 0
        
        if total_memory_gb <= 16:
            print(f"Limited total memory ({total_memory_gb:.1f} GB).")
            print(f"   Using num_workers=0 for memory efficiency.")
            return 0
        
        if available_memory_gb < 4.0:
            print(f"Low available memory ({available_memory_gb:.2f} GB < 4 GB).")
            print(f"   Using num_workers=0 to avoid memory issues.")
            return 0
        
        optimal_workers = min(2, max(1, cpu_count // 4))
        print(f"✓ Auto-detected optimal num_workers: {optimal_workers}")
        
        return optimal_workers
        
    except Exception as e:
        print(f"Could not auto-detect optimal workers: {e}")
        print(f"Defaulting to num_workers=0 for safety")
        return 0


def setup_callbacks(args, file_base) -> tuple:
    """Setup training callbacks."""
    callbacks = []
    
    viz_callback = MetricsVisualizationCallback(
        plot_every_n_epochs=args.plot_every_n_epochs,
        save_dir=os.path.join(args.log_dir, "plots"),
        file_base=file_base
    )
    callbacks.append(viz_callback)
    
    # Model Checkpoint - saves best model based on validation accuracy
    # Include model name in checkpoint filename for easy identification
    model_prefix = args.model_name.replace("/", "-").replace("apple-", "").replace("microsoft-", "")
    year_name = args.year_name
    exp_prefix = args.experiment_name or f"{model_prefix}_{year_name}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{file_base}-best-acc-epoch-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    callbacks.append(checkpoint_callback)
    
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            patience=args.early_stopping_patience,
            mode="max",
            verbose=True,
            min_delta=0.001,
        )
        callbacks.append(early_stop_callback)
    
    lr_monitor = CustomLearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    progress_bar = RichProgressBar(leave=True)
    callbacks.append(progress_bar)
    
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    return callbacks, checkpoint_callback


def setup_loggers(args, file_base) -> list:
    """Setup training loggers."""
    loggers = []
    
    tb_logger = TensorBoardLogger(save_dir=Path(args.log_dir) / "tensorboard", name=file_base,
        version="",
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    csv_logger = CSVLogger(save_dir=Path(args.log_dir) / "csv", name=file_base,
        version="",
    )
    loggers.append(csv_logger)
    
    return loggers


def train(args):
    """Main training function."""
    print("=" * 80)
    print("CLEAR-100 MULTI-CLASS CLASSIFICATION TRAINING")
    print("Using MobileViT with PyTorch Lightning")
    print("=" * 80)
    
    start_time = time.time()
    
    pl.seed_everything(args.seed, workers=True)
    print(f"\nRandom seed set to: {args.seed}")
    
    # Parse devices string (supports GPU selection like "0", "1", "0,1", etc.)
    args.devices = parse_devices(args.devices)
    
    torch.set_float32_matmul_precision('medium')
    
    if args.num_workers == -1:
        print(f"\nAuto-detecting optimal number of data loading workers...")
        # Due to shared memory limitations in some environments, default to 0
        # This avoids "Bus error" issues with multiprocessing
        args.num_workers = 0
        print(f"Setting num_workers=0 to avoid shared memory issues")
        print(f"   (Use --num_workers N to override if you have sufficient shared memory)")
    else:
        print(f"\nUsing manually specified num_workers: {args.num_workers}")
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir, "plots").mkdir(parents=True, exist_ok=True)
    
    print(f"\nInitializing data module...")
    print(f"  - Dataset root: {args.data_root}")
    print(f"  - Year: {args.year_name}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Num workers: {args.num_workers}")
    print(f"  - Augmentation: {args.augmentation}")
    
    data_module = CLEAR100DataModule(
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
    print(f"\nDataset loaded successfully!")
    print(f"  - Year: {stats['year_name']}")
    print(f"  - Number of classes: {stats['num_classes']}")
    print(f"  - Training samples: {stats['train_size']}")
    print(f"  - Validation samples: {stats['val_size']}")
    print(f"  - Test samples: {stats['test_size']}")
    
    print(f"\nInitializing model...")

    model = CLEAR100Classifier(
        model_name=args.model_name,
        num_classes=30, 
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout_rate,
        class_names=data_module.CLASS_NAMES,
    )
    
    # ========================================================================
    # FILE NAMING SETUP - All outputs will use model+year naming
    # ========================================================================
    year_name = args.year_name
    model_prefix = args.model_name.replace("/", "-").replace("apple-", "").replace("microsoft-", "")
    file_base = f"{model_prefix}_{year_name}"
    
    print("\n" +  "="*80)
    print("FILE NAMING CONFIGURATION")
    print("="*80)
    print(f"Model prefix:  {model_prefix}")
    print(f"Year:        {year_name}")
    print(f"File base:     {file_base}")
    print(f"")
    print(f"All outputs will use this prefix for clear identification:")
    print(f"  - Checkpoints:  {file_base}-best-acc-epoch-XX-val_acc-0.XXXX.ckpt")
    print(f"  - Metrics:      {file_base}_metrics.json")
    print(f"  - Summary:      {file_base}_training_summary.txt")
    print(f"  - TensorBoard:  logs/tensorboard/{file_base}/")
    print(f"  - CSV Logs:     logs/csv/{file_base}/")
    print("="*80 + "\n")
    
    print(f"\nSetting up callbacks...")
    callbacks, checkpoint_callback = setup_callbacks(args, file_base)
    
    print(f"\nSetting up loggers...")
    loggers = setup_loggers(args, file_base)
    
    print(f"\nInitializing trainer...")
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
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80 + "\n")
    
    try:
        trainer.fit(model, data_module)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        raise
    
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTraining Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total Epochs Completed: {trainer.current_epoch + 1}")
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nBest Model (by accuracy):")
        print(f"   Path: {best_model_path}")
    
    print(f"\nBest Validation Metrics:")
    print(f"   Best Accuracy: {model.best_val_acc:.4f} ({model.best_val_acc*100:.2f}%)")
    print(f"   Best F1-Score: {model.best_val_f1:.4f}")
    
    # Run test evaluation
    print("\n" + "=" * 80)
    print("RUNNING TEST EVALUATION...")
    print("=" * 80)
    
    test_results = trainer.test(model, data_module, ckpt_path=best_model_path, verbose=False)
    
    if test_results:
        test_metrics = test_results[0]
        
        # Save results to JSON
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
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80 + "\n")
    
    return trainer, model


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileViT on CLEAR-100 dataset for multi-class classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="../../datasets/CLEAR100_30classes",
                       help="Root directory of CLEAR100 dataset")
    parser.add_argument("--year_name", type=str, default="year_1",
                       help="Year to train on (e.g., year_1, year_1-2)")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="apple/mobilevit-xx-small",
                       help="HuggingFace model name")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                       help="Dropout rate")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Freeze backbone initially")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=-1,
                       help="Number of workers (-1 for auto)")
    parser.add_argument("--max_epochs", type=int, default=15,
                       help="Maximum epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                       help="Warmup epochs")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                       help="Gradient clipping")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Gradient accumulation")
    parser.add_argument("--augmentation", action="store_true", default=True,
                       help="Use augmentation")
    parser.add_argument("--no_augmentation", action="store_false", dest="augmentation")
    
    # Visualization
    parser.add_argument("--plot_every_n_epochs", type=int, default=5,
                       help="Plot frequency")
    
    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto",
                       choices=["auto", "cpu", "gpu", "tpu", "mps"])
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="GPU device(s) to use. Examples: '1' (use 1 GPU), '0' (use GPU 0), '0,1' (use GPUs 0 and 1), '2,3' (use GPUs 2 and 3)"
    )
    parser.add_argument("--precision", type=str, default="16-mixed",
                       choices=["32", "16-mixed", "bf16-mixed"])
    
    # Callbacks
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="../../../results/clear100_30classes/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="../../../results/clear100_30classes/model_checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    
    # Other
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    
    args = parser.parse_args()
    
    print("\nTraining Configuration:")
    print("-" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:30s}: {value}")
    print("-" * 80 + "\n")
    
    trainer, model = train(args)
    return trainer, model


if __name__ == "__main__":
    main()