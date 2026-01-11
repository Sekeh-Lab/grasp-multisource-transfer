"""
train_yearbook.py - Training script for Yearbook decade classification

This script orchestrates the training process for the Yearbook dataset, including
data loading, model initialization, callbacks, logging, and the training loop.
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
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import re

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

from data_module_yearbook import YearbookDataModule
from model_yearbook import YearbookDecadeClassifier
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


# ============================================================================
# RESULTS VIEWING FUNCTIONALITY
# ============================================================================

def parse_metrics_from_txt(txt_path: Path) -> dict:
    """
    Parse metrics from a checkpoint's .txt file.
    """
    metrics = {
        'validation': {},
        'test': {},
        'checkpoint_path': None,
        'subset': None,
        'experiment': None,
        'class_names': None
    }
    
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
        
        # Extract subset and experiment
        subset_match = re.search(r'Subset:\s*(\S+)', content)
        if subset_match:
            metrics['subset'] = subset_match.group(1)
        
        exp_match = re.search(r'Experiment:\s*(.+)', content)
        if exp_match:
            metrics['experiment'] = exp_match.group(1).strip()
        
        # Extract checkpoint path
        ckpt_match = re.search(r'Checkpoint:\s*(.+\.ckpt)', content)
        if ckpt_match:
            metrics['checkpoint_path'] = ckpt_match.group(1).strip()
        
        # Extract class names (decades)
        class_match = re.search(r'Decade classes:\s*\[\'([^\']+)\',\s*\'([^\']+)\'\]', content)
        if class_match:
            metrics['class_names'] = [class_match.group(1), class_match.group(2)]
        
        # Parse validation metrics
        val_section = re.search(r'VALIDATION METRICS:.*?(?=-{80}|TEST METRICS:|={80})', content, re.DOTALL)
        if val_section:
            val_text = val_section.group(0)
            metrics['validation'] = parse_metrics_section(val_text)
        
        # Parse test metrics
        test_section = re.search(r'TEST METRICS:.*?(?=-{80}|={80}|$)', content, re.DOTALL)
        if test_section:
            test_text = test_section.group(0)
            metrics['test'] = parse_metrics_section(test_text)
        
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
    
    return metrics


def parse_metrics_section(text: str) -> dict:
    """Parse a metrics section from text."""
    metrics = {}
    
    # Parse numeric metrics
    for line in text.split('\n'):
        # Match lines like "  accuracy            : 0.9823 (98.23%)"
        match = re.search(r'(\w+)\s*:\s*([\d.]+)', line)
        if match:
            key = match.group(1)
            value = float(match.group(2))
            metrics[key] = value
    
    # Parse confusion matrix
    tn_match = re.search(r'True Negatives.*?:\s*(\d+)', text)
    fp_match = re.search(r'False Positives.*?:\s*(\d+)', text)
    fn_match = re.search(r'False Negatives.*?:\s*(\d+)', text)
    tp_match = re.search(r'True Positives.*?:\s*(\d+)', text)
    
    if all([tn_match, fp_match, fn_match, tp_match]):
        metrics['confusion_matrix'] = {
            'tn': int(tn_match.group(1)),
            'fp': int(fp_match.group(1)),
            'fn': int(fn_match.group(1)),
            'tp': int(tp_match.group(1))
        }
    
    return metrics


def load_all_results(checkpoint_dir: Path) -> dict:
    """
    Load all results from .txt files in the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        Dictionary mapping subset names to their results
    """
    results = {}
    
    # Find all .txt files that accompany .ckpt checkpoints
    txt_files = list(checkpoint_dir.glob("*-best-acc-*.txt"))
    
    if not txt_files:
        return results
    
    # Load each file
    for txt_file in txt_files:
        try:
            data = parse_metrics_from_txt(txt_file)
            subset = data.get('subset', 'unknown')
            if subset != 'unknown':
                results[subset] = data
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    
    return results


def print_results_summary(checkpoint_dir: str = "./model_checkpoints"):
    """
    Print a summary of all training results from checkpoint .txt files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        print(f"   Please run training first or check the path.")
        return
    
    # Load results
    results = load_all_results(checkpoint_dir)
    
    if not results:
        print(f"No checkpoint results found in {checkpoint_dir}")
        print(f"   Please run training first.")
        print(f"   Looking for files matching: *-best-acc-*.txt")
        return
    
    # Print summary
    print("\n" + "=" * 100)
    print("YEARBOOK DECADE CLASSIFICATION - ALL RESULTS SUMMARY")
    print("=" * 100)
    print()
    
    # Sort by subset name
    sorted_subsets = sorted(results.keys())
    
    # Print table header
    print(f"{'Subset':<20} {'Decades':<20} {'Split':<12} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUROC':>8} {'Loss':>8}")
    print("-" * 120)
    
    # Print results for each subset
    for subset in sorted_subsets:
        data = results[subset]
        class_names = data.get('class_names', ['Decade1', 'Decade2'])
        decade_str = f"{class_names[0]} vs {class_names[1]}"
        
        # Validation results
        if 'validation' in data and data['validation']:
            val = data['validation']
            print(f"{subset:<20} {decade_str:<20} {'Validation':<12} "
                  f"{val.get('accuracy', 0)*100:7.2f}% "
                  f"{val.get('f1_score', 0):7.4f} "
                  f"{val.get('precision', 0):7.4f} "
                  f"{val.get('recall', 0):7.4f} "
                  f"{val.get('auroc', 0):7.4f} "
                  f"{val.get('loss', 0):7.4f}")
        
        # Test results
        if 'test' in data and data['test']:
            test = data['test']
            print(f"{'':<20} {'':<20} {'Test':<12} "
                  f"{test.get('accuracy', 0)*100:7.2f}% "
                  f"{test.get('f1_score', 0):7.4f} "
                  f"{test.get('precision', 0):7.4f} "
                  f"{test.get('recall', 0):7.4f} "
                  f"{test.get('auroc', 0):7.4f} "
                  f"{test.get('loss', 0):7.4f}")
        
        print()
    
    print("=" * 120)
    print()
    
    # Print detailed confusion matrices
    print("CONFUSION MATRICES (Test Set):")
    print("=" * 100)
    print()
    
    for subset in sorted_subsets:
        data = results[subset]
        class_names = data.get('class_names', ['Decade1', 'Decade2'])
        
        if 'test' in data and data['test'] and 'confusion_matrix' in data['test']:
            cm = data['test']['confusion_matrix']
            tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
            
            print(f"{subset} ({class_names[0]} vs {class_names[1]}):")
            print(f"                Predicted {class_names[0]:<8s}  Predicted {class_names[1]:<8s}")
            print(f"    Actual {class_names[0]:<8s}: {tn:7d}      {fp:7d}")
            print(f"    Actual {class_names[1]:<8s}: {fn:7d}      {tp:7d}")
            print()
    
    print("=" * 100)
    print()
    
    # Print best performing subset
    best_subset = None
    best_acc = 0.0
    
    for subset, data in results.items():
        if 'test' in data and data['test']:
            acc = data['test'].get('accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best_subset = subset
    
    if best_subset:
        best_data = results[best_subset]
        class_names = best_data.get('class_names', ['Decade1', 'Decade2'])
        print(f"Best Performing Subset: {best_subset}")
        print(f"   Decades: {class_names[0]} vs {class_names[1]}")
        print(f"   Test Accuracy: {best_acc*100:.2f}%")
        ckpt_path = best_data.get('checkpoint_path', 'N/A')
        if ckpt_path:
            print(f"   Checkpoint: {ckpt_path}")
        print()
    
    # List all .ckpt files found
    print("=" * 100)
    print("CHECKPOINT FILES:")
    print("=" * 100)
    for subset in sorted_subsets:
        data = results[subset]
        ckpt = data.get('checkpoint_path', 'N/A')
        if ckpt and ckpt != 'N/A':
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                print(f"  {subset:20s}: {ckpt_path.name} ({size_mb:.1f} MB)")
            else:
                # Try relative path
                rel_ckpt = checkpoint_dir / ckpt_path.name
                if rel_ckpt.exists():
                    size_mb = rel_ckpt.stat().st_size / (1024 * 1024)
                    print(f"  {subset:20s}: {rel_ckpt.name} ({size_mb:.1f} MB)")
                else:
                    print(f"  {subset:20s}: {ckpt_path.name} (not found)")
    print()


# ============================================================================
# TRAINING CALLBACKS
# ============================================================================


class CustomLearningRateMonitor(Callback):
    """
    Custom learning rate monitor that avoids the torch.tensor() warning.
    Logs learning rates for all parameter groups.
    """
    
    def __init__(self, logging_interval='step'):
        super().__init__()
        self.logging_interval = logging_interval
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Log learning rate at the start of each training batch."""
        if self.logging_interval == 'step':
            self._log_lr(trainer, pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Log learning rate at the start of each epoch."""
        if self.logging_interval == 'epoch':
            self._log_lr(trainer, pl_module)
    
    def _log_lr(self, trainer, pl_module):
        """Extract and log learning rates without triggering tensor warnings."""
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                
                # Convert to float directly without torch.tensor()
                if isinstance(lr, torch.Tensor):
                    lr_value = float(lr.detach().cpu().item())
                else:
                    lr_value = float(lr)
                
                # Log with appropriate name
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
        
        # Storage for metrics
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        current_epoch = trainer.current_epoch
        
        # Get metrics from callback_metrics
        if trainer.callback_metrics:
            # Get and store losses
            train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
            val_loss = trainer.callback_metrics.get('val_loss', None)
            
            # Get and store accuracies
            train_acc = trainer.callback_metrics.get('train_acc', None)
            val_acc = trainer.callback_metrics.get('val_acc', None)
            
            # Only add to history if we have all required metrics
            if train_loss is not None and val_loss is not None and train_acc is not None and val_acc is not None:
                train_loss_val = float(train_loss.item() if torch.is_tensor(train_loss) else train_loss)
                val_loss_val = float(val_loss.item() if torch.is_tensor(val_loss) else val_loss)
                train_acc_val = float(train_acc.item() if torch.is_tensor(train_acc) else train_acc)
                val_acc_val = float(val_acc.item() if torch.is_tensor(val_acc) else val_acc)
                
                # Append all metrics together to keep them synchronized
                self.epochs.append(current_epoch)
                self.train_losses.append(train_loss_val)
                self.val_losses.append(val_loss_val)
                self.train_accs.append(train_acc_val)
                self.val_accs.append(val_acc_val)
        
        # Plot every N epochs if we have data
        if (current_epoch + 1) % self.plot_every_n_epochs == 0 and len(self.epochs) > 0:
            self._plot_metrics(current_epoch)
            # Force garbage collection after plotting
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _plot_metrics(self, current_epoch):
        """Plot and save metrics."""
        try:
            # Verify all lists have the same length
            if not (len(self.epochs) == len(self.train_losses) == len(self.val_losses) == 
                    len(self.train_accs) == len(self.val_accs)):
                print(f"Skipping plot - metric lists have different lengths")
                return
            
            # Need at least 2 points to plot
            if len(self.epochs) < 2:
                print(f"Skipping plot - need at least 2 data points, have {len(self.epochs)}")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot Loss
            ax1.plot(self.epochs, self.train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
            ax1.plot(self.epochs, self.val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax1.set_title(f'Training and Validation Loss (Epoch {current_epoch + 1})', 
                         fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(left=0)
            ax1.set_ylim(bottom=0)
            
            # Plot Accuracy (as percentages)
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
            ax2.set_xlim(left=0)
            ax2.set_ylim([0, 100])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.save_dir / f"{self.file_base}_metrics_epoch_{current_epoch + 1:03d}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            print(f"\nMetrics plot saved to: {plot_path}")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
        finally:
            plt.close('all')
    
    def on_train_end(self, trainer, pl_module):
        """Create final plot at the end of training."""
        if len(self.epochs) > 0:
            self._plot_metrics(trainer.current_epoch)
            print(f"Final metrics plot saved")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    return {}


def get_optimal_num_workers() -> int:
    """
    Automatically determine optimal number of data loading workers based on system resources.
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
        print(f"Auto-detected optimal num_workers: {optimal_workers}")
        
        return optimal_workers
        
    except Exception as e:
        print(f"Could not auto-detect optimal workers: {e}")
        print(f"Defaulting to num_workers=0 for safety")
        return 0


def setup_callbacks(args, file_base) -> tuple:
    """Setup training callbacks. Returns (callbacks_list, checkpoint_callback)."""
    callbacks = []
    
    # Metrics Visualization Callback
    viz_callback = MetricsVisualizationCallback(
        plot_every_n_epochs=args.plot_every_n_epochs,
        save_dir=os.path.join(args.log_dir, "plots"),
        file_base=file_base
    )
    callbacks.append(viz_callback)
    
    # Model Checkpoint - saves best model based on validation accuracy
    # Include model name in checkpoint filename for easy identification
    model_prefix = args.model_name.replace("/", "-").replace("apple-", "").replace("microsoft-", "")
    subset_name = args.subset_name
    exp_prefix = args.experiment_name or f"{model_prefix}_{subset_name}"
    
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
    
    # Early Stopping
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            patience=args.early_stopping_patience,
            mode="max",
            verbose=True,
            min_delta=0.001,
        )
        callbacks.append(early_stop_callback)
    
    # Learning Rate Monitor
    lr_monitor = CustomLearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Rich Progress Bar
    progress_bar = RichProgressBar(leave=True)
    callbacks.append(progress_bar)
    
    # Rich Model Summary
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    return callbacks, checkpoint_callback


def setup_loggers(args, file_base) -> list:
    """Setup training loggers."""
    loggers = []
    
    # TensorBoard Logger
    tb_logger = TensorBoardLogger(save_dir=Path(args.log_dir) / "tensorboard", name=file_base,
        version="",
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # CSV Logger
    csv_logger = CSVLogger(save_dir=Path(args.log_dir) / "csv", name=file_base,
        version="",
    )
    loggers.append(csv_logger)
    
    return loggers


def calculate_class_weights(data_module: YearbookDataModule) -> Optional[list]:
    """
    Calculate class weights based on training data distribution.
    """
    stats = data_module.get_dataset_stats()
    train_dist = stats.get('train_class_distribution', {})
    class_names = stats.get('class_names', ['Decade1', 'Decade2'])
    
    if not train_dist or len(train_dist) != 2:
        return None
    
    total = sum(train_dist.values())
    decade1_count = train_dist.get(class_names[0], total / 2)
    decade2_count = train_dist.get(class_names[1], total / 2)
    
    # Calculate inverse frequency weights
    decade1_weight = total / (2 * decade1_count)
    decade2_weight = total / (2 * decade2_count)
    
    print(f"\nClass distribution:")
    print(f"  {class_names[0]}: {decade1_count} samples (weight: {decade1_weight:.4f})")
    print(f"  {class_names[1]}: {decade2_count} samples (weight: {decade2_weight:.4f})")
    
    # Only use weights if imbalance is significant (>55/45)
    imbalance_ratio = max(decade1_count, decade2_count) / min(decade1_count, decade2_count)
    if imbalance_ratio > 1.22:
        print(f"  Imbalance ratio: {imbalance_ratio:.2f} - using class weights")
        return [decade1_weight, decade2_weight]
    else:
        print(f"  Imbalance ratio: {imbalance_ratio:.2f} - classes are balanced")
        return None


def plot_confusion_matrix(tn, fp, fn, tp, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix as a heatmap.
    """
    try:
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'},
                    annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Actual Decade', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Decade', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
    finally:
        plt.close('all')


def save_checkpoint_metrics_file(ckpt_path: str, all_metrics: dict, args, class_names: list):
    """
    Save the metrics file for a checkpoint with complete metrics.
    """
    if not ckpt_path:
        return
    
    ckpt_path = Path(ckpt_path)
    
    # Create .txt file
    if ckpt_path.suffix == '.ckpt':
        metrics_path = ckpt_path.with_suffix('.txt')
    else:
        metrics_path = Path(str(ckpt_path) + '.txt')
    
    with open(metrics_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"BEST CHECKPOINT METRICS: {ckpt_path.name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Subset: {args.subset_name}\n")
        f.write(f"Experiment: {args.experiment_name or args.subset_name}\n")
        f.write(f"Decade classes: {class_names}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Validation metrics
        if 'validation' in all_metrics and all_metrics['validation']:
            f.write("-" * 80 + "\n")
            f.write("VALIDATION METRICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in all_metrics['validation'].items():
                if key == 'confusion_matrix':
                    cm = value
                    f.write(f"\nConfusion Matrix:\n")
                    f.write(f"  True Negatives ({class_names[0]}):  {cm['tn']}\n")
                    f.write(f"  False Positives ({class_names[1]}): {cm['fp']}\n")
                    f.write(f"  False Negatives ({class_names[0]}): {cm['fn']}\n")
                    f.write(f"  True Positives ({class_names[1]}):  {cm['tp']}\n")
                else:
                    f.write(f"  {key:20s}: {value:.4f}")
                    if 'accuracy' in key:
                        f.write(f" ({value*100:.2f}%)")
                    f.write("\n")
            f.write("\n")
        
        # Test metrics
        if 'test' in all_metrics and all_metrics['test']:
            f.write("-" * 80 + "\n")
            f.write("TEST METRICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in all_metrics['test'].items():
                if key == 'confusion_matrix':
                    cm = value
                    f.write(f"\nConfusion Matrix:\n")
                    f.write(f"  True Negatives ({class_names[0]}):  {cm['tn']}\n")
                    f.write(f"  False Positives ({class_names[1]}): {cm['fp']}\n")
                    f.write(f"  False Negatives ({class_names[0]}): {cm['fn']}\n")
                    f.write(f"  True Positives ({class_names[1]}):  {cm['tp']}\n")
                else:
                    f.write(f"  {key:20s}: {value:.4f}")
                    if 'accuracy' in key:
                        f.write(f" ({value*100:.2f}%)")
                    f.write("\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("HOW TO LOAD THIS MODEL:\n")
        f.write("=" * 80 + "\n\n")
        f.write("Python code:\n")
        f.write("-" * 40 + "\n")
        f.write("from model import YearbookDecadeClassifier\n\n")
        f.write("# Method 1: Using PyTorch Lightning's load_from_checkpoint (Recommended)\n")
        f.write("model = YearbookDecadeClassifier.load_from_checkpoint(\n")
        f.write(f"    '{ckpt_path.name}',\n")
        f.write(f"    model_name='{args.model_name}',\n")
        f.write("    num_classes=2,\n")
        f.write(f"    class_names={class_names}\n")
        f.write(")\n")
        f.write("model.eval()\n\n")
        f.write("# Method 2: Manual loading\n")
        f.write("import torch\n")
        f.write(f"checkpoint = torch.load('{ckpt_path.name}')\n")
        f.write("model = YearbookDecadeClassifier(\n")
        f.write(f"    model_name='{args.model_name}',\n")
        f.write("    num_classes=2,\n")
        f.write(f"    class_names={class_names}\n")
        f.write(")\n")
        f.write("model.load_state_dict(checkpoint['state_dict'])\n")
        f.write("model.eval()\n")
    
    print(f"Saved metrics file: {metrics_path}")


def print_training_summary(trainer, model, start_time, args):
    """
    Print comprehensive training summary with all metrics.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Trained model
        start_time: Training start time
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    # Training duration
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"\nTraining Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total Epochs Completed: {trainer.current_epoch + 1}")
    
    # Best checkpoint information
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    
    if best_model_path:
        print(f"\nBest Model (by accuracy):")
        print(f"   Path: {best_model_path}")
        if best_score is not None:
            print(f"   Validation Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
    
    # Best metrics from model
    print(f"\nBest Validation Metrics:")
    print(f"   Best Accuracy: {model.best_val_acc:.4f} ({model.best_val_acc*100:.2f}%)")
    print(f"   Best F1-Score: {model.best_val_f1:.4f}")
    
    # Log locations
    print(f"\nLogs and Checkpoints:")
    print(f"   Checkpoints: {args.checkpoint_dir}")
    print(f"   Logs: {args.log_dir}")
    print(f"   Plots: {os.path.join(args.log_dir, 'plots')}")
    print(f"   TensorBoard: tensorboard --logdir {args.log_dir}")
    
    return best_model_path


def run_final_evaluation(trainer, model, data_module, best_model_path, args, file_base):
    """
    Run comprehensive final evaluation on validation and test sets using the best checkpoint.
    """
    print("\n" + "=" * 80)
    print("FINAL EVALUATION - USING BEST CHECKPOINT")
    print("=" * 80)
    print(f"Loading best model from: {best_model_path}")
    
    # Get class names
    class_names = data_module.class_names
    
    # Create results directory
    results_dir = Path(args.checkpoint_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots directory
    plots_dir = Path(args.log_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation set evaluation with best model
    print(f"\nValidation Set Performance (Best Model):")
    print(f"    Decades: {class_names[0]} vs {class_names[1]}")
    print("-" * 80)
    val_results = trainer.validate(model, data_module, ckpt_path=best_model_path, verbose=False)
    
    val_metrics_dict = {}
    if val_results:
        val_metrics = val_results[0]
        val_acc = val_metrics.get('val_acc', 0)
        val_f1 = val_metrics.get('val_f1', 0)
        val_precision = val_metrics.get('val_precision', 0)
        val_recall = val_metrics.get('val_recall', 0)
        val_auroc = val_metrics.get('val_auroc', 0)
        val_loss = val_metrics.get('val_loss', 0)
        
        print(f"\n  Loss:      {val_loss:.4f}")
        print(f"  Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  F1-Score:  {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  AUROC:     {val_auroc:.4f}")
        
        val_metrics_dict = {
            'loss': float(val_loss),
            'accuracy': float(val_acc),
            'f1_score': float(val_f1),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'auroc': float(val_auroc)
        }
        
        # Confusion matrix
        tn = val_metrics.get('val_tn', 0)
        fp = val_metrics.get('val_fp', 0)
        fn = val_metrics.get('val_fn', 0)
        tp = val_metrics.get('val_tp', 0)
        
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted {class_names[0]:<8s}  Predicted {class_names[1]:<8s}")
        print(f"    Actual {class_names[0]:<8s}: {tn:7.0f}      {fp:7.0f}")
        print(f"    Actual {class_names[1]:<8s}: {fn:7.0f}      {tp:7.0f}")
        
        val_metrics_dict['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
        
        # Plot confusion matrix
        plot_confusion_matrix(tn, fp, fn, tp, class_names,
                            title=f"Validation Set Confusion Matrix - {args.subset_name}",
                            save_path=plots_dir / f"{file_base}_confusion_matrix_val.png")
        
        # Per-class metrics
        if tn + fp > 0:
            decade1_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            decade1_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"\n  {class_names[0]} Performance:")
            print(f"    Precision: {decade1_precision:.4f} ({decade1_precision*100:.2f}%)")
            print(f"    Recall:    {decade1_recall:.4f} ({decade1_recall*100:.2f}%)")
        
        if tp + fn > 0:
            decade2_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            decade2_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"\n  {class_names[1]} Performance:")
            print(f"    Precision: {decade2_precision:.4f} ({decade2_precision*100:.2f}%)")
            print(f"    Recall:    {decade2_recall:.4f} ({decade2_recall*100:.2f}%)")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test set evaluation
    print("\n" + "=" * 80)
    print(f"Test Set Performance (Best Model):")
    print(f"    Decades: {class_names[0]} vs {class_names[1]}")
    print("-" * 80)
    
    test_results = trainer.test(model, data_module, ckpt_path=best_model_path, verbose=False)
    
    test_metrics_dict = {}
    if test_results:
        test_metrics = test_results[0]
        test_acc = test_metrics.get('test_acc', 0)
        test_f1 = test_metrics.get('test_f1', 0)
        test_precision = test_metrics.get('test_precision', 0)
        test_recall = test_metrics.get('test_recall', 0)
        test_auroc = test_metrics.get('test_auroc', 0)
        test_loss = test_metrics.get('test_loss', 0)
        
        print(f"\n  Loss:      {test_loss:.4f}")
        print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  F1-Score:  {test_f1:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print(f"  AUROC:     {test_auroc:.4f}")
        
        test_metrics_dict = {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'f1_score': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'auroc': float(test_auroc)
        }
        
        # Test confusion matrix
        tn = test_metrics.get('test_tn', 0)
        fp = test_metrics.get('test_fp', 0)
        fn = test_metrics.get('test_fn', 0)
        tp = test_metrics.get('test_tp', 0)
        
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted {class_names[0]:<8s}  Predicted {class_names[1]:<8s}")
        print(f"    Actual {class_names[0]:<8s}: {tn:7.0f}      {fp:7.0f}")
        print(f"    Actual {class_names[1]:<8s}: {fn:7.0f}      {tp:7.0f}")
        
        test_metrics_dict['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
        
        plot_confusion_matrix(tn, fp, fn, tp, class_names,
                            title=f"Test Set Confusion Matrix - {args.subset_name}",
                            save_path=plots_dir / f"{file_base}_confusion_matrix_test.png")
    
    # Save metrics to JSON file
    import json
    exp_name = args.experiment_name or args.subset_name
    metrics_file = results_dir / f"{exp_name}_best_model_metrics.json"
    
    all_metrics = {
        'subset': args.subset_name,
        'experiment_name': exp_name,
        'class_names': class_names,
        'best_checkpoint': str(best_model_path),
        'validation': val_metrics_dict,
        'test': test_metrics_dict,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nBest model metrics saved to: {metrics_file}")
    
    # Also save a simple text summary
    summary_file = results_dir / f"{exp_name}_best_model_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"BEST MODEL RESULTS - {args.subset_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Decades: {class_names[0]} vs {class_names[1]}\n")
        f.write(f"Best Checkpoint: {best_model_path}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("VALIDATION SET RESULTS:\n")
        f.write("-" * 80 + "\n")
        for key, value in val_metrics_dict.items():
            if key != 'confusion_matrix':
                f.write(f"  {key:15s}: {value:.4f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("TEST SET RESULTS:\n")
        f.write("-" * 80 + "\n")
        for key, value in test_metrics_dict.items():
            if key != 'confusion_matrix':
                f.write(f"  {key:15s}: {value:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Best model summary saved to: {summary_file}")
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_metrics


def save_training_summary(args, stats, best_model_path, duration, final_metrics=None):
    """
    Save training summary to a text file.
    """
    exp_name = args.experiment_name or args.subset_name
    summary_path = Path(args.checkpoint_dir) / f"{exp_name}_training_summary.txt"
    
    class_names = stats.get('class_names', ['Decade1', 'Decade2'])
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("YEARBOOK DECADE CLASSIFICATION - TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Subset: {args.subset_name}\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Decades: {class_names[0]} vs {class_names[1]}\n")
        f.write(f"Model: {args.model_name}\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training samples:   {stats.get('train_size', 0)}\n")
        f.write(f"Validation samples: {stats.get('val_size', 0)}\n")
        f.write(f"Test samples:       {stats.get('test_size', 0)}\n")
        f.write(f"Class distribution: {stats.get('train_class_distribution', {})}\n\n")
        
        f.write("Training Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Batch size:         {args.batch_size}\n")
        f.write(f"Learning rate:      {args.learning_rate}\n")
        f.write(f"Max epochs:         {args.max_epochs}\n")
        f.write(f"Weight decay:       {args.weight_decay}\n")
        f.write(f"Warmup epochs:      {args.warmup_epochs}\n")
        f.write(f"Dropout rate:       {args.dropout_rate}\n")
        f.write(f"Augmentation:       {args.augmentation}\n")
        f.write(f"Freeze backbone:    {args.freeze_backbone}\n\n")
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        f.write(f"Training duration:  {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")
        
        f.write(f"Best model path: {best_model_path}\n\n")
        
        # Add final metrics if available
        if final_metrics:
            f.write("=" * 80 + "\n")
            f.write("BEST MODEL PERFORMANCE:\n")
            f.write("=" * 80 + "\n\n")
            
            if 'test' in final_metrics and final_metrics['test']:
                f.write("Test Set Results:\n")
                f.write("-" * 80 + "\n")
                for key, value in final_metrics['test'].items():
                    if key != 'confusion_matrix':
                        f.write(f"  {key:15s}: {value:.4f}\n")
                f.write("\n")
            
            if 'validation' in final_metrics and final_metrics['validation']:
                f.write("Validation Set Results:\n")
                f.write("-" * 80 + "\n")
                for key, value in final_metrics['validation'].items():
                    if key != 'confusion_matrix':
                        f.write(f"  {key:15s}: {value:.4f}\n")
    
    print(f"\nTraining summary saved to: {summary_path}")


def train(args):
    """Main training function."""
    print("=" * 80)
    print("YEARBOOK DECADE CLASSIFICATION TRAINING")
    print("Using MobileViT with PyTorch Lightning")
    print("=" * 80)
    
    start_time = time.time()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    print(f"\nRandom seed set to: {args.seed}")
    
    # Parse devices string (supports GPU selection like "0", "1", "0,1", etc.)
    args.devices = parse_devices(args.devices)
    
    # Set tensor precision for better performance
    torch.set_float32_matmul_precision('medium')
    print(f"Float32 matmul precision set to 'medium' for better performance")
    
    # Auto-detect optimal num_workers if set to -1
    if args.num_workers == -1:
        print(f"\nAuto-detecting optimal number of data loading workers...")
        args.num_workers = get_optimal_num_workers()
    else:
        print(f"\nUsing manually specified num_workers: {args.num_workers}")
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir, "plots").mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    print(f"\nInitializing data module...")
    print(f"  - Dataset root: {args.data_root}")
    print(f"  - Subset: {args.subset_name}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Num workers: {args.num_workers}")
    print(f"  - Augmentation: {args.augmentation}")
    
    data_module = YearbookDataModule(
        data_root=args.data_root,
        subset_name=args.subset_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
    )
    
    # Setup data module to get statistics
    data_module.prepare_data()
    data_module.setup()
    
    # Print dataset statistics
    stats = data_module.get_dataset_stats()
    class_names = stats['class_names']
    print(f"\nDataset loaded successfully!")
    print(f"  - Decades: {class_names[0]} vs {class_names[1]}")
    print(f"  - Training samples: {stats['train_size']}")
    print(f"  - Validation samples: {stats['val_size']}")
    print(f"  - Test samples: {stats['test_size']}")
    
    # Calculate class weights if needed
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(data_module)
    
    # Initialize model
    print(f"\nInitializing model...")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"  - Warmup epochs: {args.warmup_epochs}")
    print(f"  - Freeze backbone: {args.freeze_backbone}")
    print(f"  - Dropout rate: {args.dropout_rate}")
    
    model = YearbookDecadeClassifier(
        model_name=args.model_name,
        num_classes=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout_rate,
        class_weights=class_weights,
        class_names=class_names,
    )
    
    # ========================================================================
    # FILE NAMING SETUP - Define file_base for all outputs
    # ========================================================================
    model_prefix = args.model_name.replace("/", "-").replace("apple-", "").replace("microsoft-", "")
    subset_name = args.subset_name
    file_base = f"{model_prefix}_{subset_name}"
    
    print("\n" + "="*80)
    print("FILE NAMING CONFIGURATION")
    print("="*80)
    print(f"Model prefix:  {model_prefix}")
    print(f"Subset:        {subset_name}")
    print(f"File base:     {file_base}")
    print(f"")
    print(f"All outputs will use this prefix for clear identification:")
    print(f"  - Checkpoints:  {file_base}-best-acc-epoch-XX-val_acc-0.XXXX.ckpt")
    print(f"  - Metrics:      {file_base}_metrics.json")
    print(f"  - Summary:      {file_base}_training_summary.txt")
    print(f"  - TensorBoard:  logs/tensorboard/{file_base}/")
    print(f"  - CSV Logs:     logs/csv/{file_base}/")
    print(f"  - Plots:        logs/plots/{file_base}_*.png")
    print("="*80 + "\n")
    
    # Setup callbacks
    print(f"Setting up callbacks...")
    callbacks, checkpoint_callback = setup_callbacks(args, file_base)
    for callback in callbacks:
        print(f"  - {callback.__class__.__name__}")
    
    # Setup loggers
    print(f"\nSetting up loggers...")
    loggers = setup_loggers(args, file_base)
    for logger in loggers:
        print(f"  - {logger.__class__.__name__}")
    
    # Initialize trainer
    print(f"\nInitializing trainer...")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Accelerator: {args.accelerator}")
    print(f"  - Devices: {args.devices}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Gradient clip: {args.gradient_clip_val}")
    print(f"  - Plot every N epochs: {args.plot_every_n_epochs}")
    
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
    
    # Resume from checkpoint
    ckpt_path = None
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        ckpt_path = args.resume_from_checkpoint
        print(f"\nResuming from checkpoint: {ckpt_path}")
    
    # Train the model
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80 + "\n")
    
    try:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise
    
    # Print training summary
    duration = time.time() - start_time
    best_model_path = print_training_summary(trainer, model, start_time, args)
    
    # Run final evaluation
    if best_model_path:
        final_metrics = run_final_evaluation(trainer, model, data_module, best_model_path, args, file_base)
        
        # Create/update the metrics file
        if final_metrics:
            save_checkpoint_metrics_file(best_model_path, final_metrics, args, class_names)
    else:
        print("\nWarning: No best model checkpoint found. Skipping final evaluation.")
        final_metrics = None
    
    # Save training summary to file
    save_training_summary(args, stats, best_model_path, duration, final_metrics)
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("ALL DONE! Training and evaluation complete.")
    print("=" * 80 + "\n")
    
    return trainer, model


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train MobileViT on Yearbook dataset for decade classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../datasets/Yearbook_Decades",
        help="Root directory of Yearbook dataset",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="before_1950s",
        choices=["before_1950s", "1950s_1960s", "1970s_1980s", "1990s_and_later"],
        help="Yearbook temporal subset to use",
    )
    
    # For compatibility with shell script
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        dest="subset_name",
        choices=["before_1950s", "1950s_1960s", "1970s_1980s", "1990s_and_later"],
        help="Alias for --subset_name (for shell script compatibility)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="apple/mobilevit-xx-small",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for classification head",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone initially",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Number of data loading workers (-1 for auto-detect, 0 to disable multiprocessing, >0 for manual setting)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=15,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation batches",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        default=True,
        help="Use data augmentation",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_false",
        dest="augmentation",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights for imbalanced data",
    )
    
    # Visualization arguments
    parser.add_argument(
        "--plot_every_n_epochs",
        type=int,
        default=5,
        help="Plot metrics every N epochs",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "tpu", "mps"],
        help="Accelerator type",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="GPU device(s) to use. Examples: '1' (use 1 GPU), '0' (use GPU 0), '0,1' (use GPUs 0 and 1), '2,3' (use GPUs 2 and 3)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision (16-mixed recommended for GPUs)",
    )
    
    # Callback arguments
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=7,
        help="Early stopping patience",
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for logs",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./model_checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume from",
    )
    
    # View results mode
    parser.add_argument(
        "--view_results",
        action="store_true",
        help="View summary of all training results and exit (no training)",
    )
    
    args = parser.parse_args()
    
    # If view_results mode, print results and exit
    if args.view_results:
        print_results_summary(args.checkpoint_dir)
        return None, None
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 80)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:30s}: {value}")
    print("-" * 80 + "\n")
    
    # Start training
    trainer, model = train(args)
    
    # trainer and model will be None if --view_results was used
    if trainer is None:
        return
    
    return trainer, model


if __name__ == "__main__":
    main()