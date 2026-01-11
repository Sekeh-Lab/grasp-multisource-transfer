#!/usr/bin/env python
"""
run_fusion_experiment.py - Unified Fusion Transfer Learning Experiment with FLOPs Tracking

Fusion transfer learning: Merge multiple source models through weighted averaging,
then fine-tune on target data.

Usage:
    # Single experiment
    python run_fusion_experiment.py \
        --dataset clear10 \
        --target year_1-2 \
        --sources year_3-4 year_5-6 year_7-8 year_9-10 \
        --trial 1
    
    # Different dataset
    python run_fusion_experiment.py \
        --dataset yearbook \
        --target before_1950s \
        --sources 1950s_1960s 1970s_1980s 1990s_and_later \
        --trial 1
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add shared utilities to path
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_utils import get_dataloaders
from model_utils import load_baseline_model, standardize_model_name
from metrics_utils import calculate_metrics, save_confusion_matrix, create_training_curves
from gpu_memory_tracker import GPUMemoryTracker

# Import FLOPs counter
try:
    from flops_counter import get_fusion_flops_report
    FLOPS_AVAILABLE = True
except ImportError:
    print("Warning: flops_counter not available. FLOPs will not be calculated.")
    FLOPS_AVAILABLE = False

class FusionClassifier(nn.Module):
    """
    Fusion ensemble that merges multiple models through weighted averaging.
    """
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        """
        Args:
            models: List of source models to fuse
            weights: Optional fusion weights (default: uniform)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, x):
        """Forward pass through all models and average predictions."""
        logits_list = []
        
        for model in self.models:
            outputs = model(x)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', list(outputs.values())[0])
            else:
                logits = outputs
            logits_list.append(logits)
        
        # Weighted average
        stacked_logits = torch.stack(logits_list)
        weighted_logits = (stacked_logits * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        return weighted_logits

def fine_tune_fusion_model(
    fusion_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01
) -> Dict:
    """Fine-tune fusion model on target data."""
    
    fusion_model = fusion_model.to(device)
    fusion_model.train()
    
    optimizer = optim.AdamW(
        fusion_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        fusion_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = fusion_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        fusion_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                logits = fusion_model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(fusion_model.state_dict())
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}% "
              f"{'BEST' if val_acc == best_val_acc else ''}")
    
    if best_state is not None:
        fusion_model.load_state_dict(best_state)
    
    return history


def evaluate_fusion_model(
    fusion_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate fusion model."""
    
    fusion_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            
            logits = fusion_model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_labels, all_preds, all_probs


def calculate_flops_for_fusion(
    single_model: nn.Module,
    n_sources: int,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    num_classes: int,
    finetune_epochs: int,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224)
) -> Dict:
    """Calculate FLOPs for fusion experiment."""
    
    if not FLOPS_AVAILABLE:
        return {
            'method': 'FUSION',
            'error': 'FLOPs counter not available',
            'total_gflops': 0.0
        }
    
    try:
        flops_report = get_fusion_flops_report(
            model=single_model,
            input_size=input_size,
            n_sources=n_sources,
            n_train_samples=n_train_samples,
            n_val_samples=n_val_samples,
            n_test_samples=n_test_samples,
            num_classes=num_classes,
            finetune_epochs=finetune_epochs,
            device=device
        )
        return flops_report
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return {
            'method': 'FUSION',
            'error': str(e),
            'total_gflops': 0.0
        }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_fusion_experiment(
    dataset: str,
    target: str,
    sources: List[str],
    args,
    device: torch.device,
    trial_num: int = 1
) -> Dict:
    """Run complete fusion transfer learning experiment."""
    
    model_prefix = standardize_model_name(args.model)
    experiment_name = f"{model_prefix}_{target}_fusion_trial{trial_num}"
    output_dir = Path(args.results_dir) / dataset / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{experiment_name}_comprehensive_report.txt"
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + "\n")
    
    experiment_start = time.perf_counter()
    
    # Initialize GPU memory tracker
    memory_tracker = GPUMemoryTracker(device)
    memory_tracker.checkpoint('experiment_start')
    
    log_print("=" * 80)
    log_print(f"FUSION TRANSFER LEARNING EXPERIMENT")
    log_print("=" * 80)
    log_print(f"\nModel: {args.model}")
    log_print(f"Dataset: {dataset}")
    log_print(f"Target: {target}")
    log_print(f"Sources: {', '.join(sources)}")
    log_print(f"Trial: {trial_num}")
    log_print(f"Device: {device}")
    log_print("=" * 80 + "\n")
    
    log_print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset=dataset,
        target=target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model
    )
    
    # Get dataset sizes
    n_train_samples = len(train_loader.dataset)
    n_val_samples = len(val_loader.dataset)
    n_test_samples = len(test_loader.dataset)
    
    log_print(f"Data loaded: {num_classes} classes")
    log_print(f"Train samples: {n_train_samples}")
    log_print(f"Validation samples: {n_val_samples}")
    log_print(f"Test samples: {n_test_samples}\n")
    memory_tracker.checkpoint('data_loaded')
    
    log_print(f"Loading {len(sources)} source models...")
    source_models = []
    
    for source in sources:
        model = load_baseline_model(
            dataset=dataset,
            subset=source,
            model_name=args.model,
            num_classes=num_classes
        )
        if model is not None:
            source_models.append(model)
            log_print(f"  Loaded model for {source}")
        else:
            log_print(f"  Failed to load model for {source}")
    
    if len(source_models) == 0:
        log_print("\nERROR: No source models loaded. Cannot create fusion model.")
        return None
    
    log_print(f"Loaded {len(source_models)}/{len(sources)} source models\n")
    memory_tracker.checkpoint('source_models_loaded')
    
    log_print("Creating fusion model...")
    fusion_model = FusionClassifier(source_models)
    log_print(f"Fusion model created (uniform weights)\n")
    memory_tracker.checkpoint('fusion_model_created')
    
    # Calculate FLOPs
    log_print("Calculating FLOPs...")
    flops_report = calculate_flops_for_fusion(
        single_model=source_models[0],  # Use first model as reference
        n_sources=len(source_models),
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        num_classes=num_classes,
        finetune_epochs=args.finetune_epochs,
        device=device,
        input_size=(1, 3, 224, 224)
    )
    
    if FLOPS_AVAILABLE and 'error' not in flops_report:
        log_print(f"FLOPs calculated successfully")
        log_print(f"  Total GFLOPs: {flops_report['total_gflops']:.2f}")
        log_print(f"  Merging GFLOPs: {flops_report['merging_gflops']:.4f}")
        log_print(f"  Training GFLOPs: {flops_report['training_gflops']:.2f}")
        log_print(f"  Inference GFLOPs: {flops_report['inference_gflops']:.2f}")
    else:
        log_print(f"FLOPs calculation skipped or failed")
    log_print("")
    
    log_print(f"Fine-tuning fusion model ({args.finetune_epochs} epochs)...")
    history = fine_tune_fusion_model(
        fusion_model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.finetune_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    log_print("Fine-tuning complete\n")
    memory_tracker.checkpoint('finetuning_complete')
    
    log_print("Evaluating on validation set...")
    val_metrics, val_labels, val_preds, val_probs = evaluate_fusion_model(
        fusion_model, val_loader, device
    )
    
    log_print("Evaluating on test set...")
    test_metrics, test_labels, test_preds, test_probs = evaluate_fusion_model(
        fusion_model, test_loader, device
    )
    
    # Get GPU memory summary
    memory_tracker.checkpoint('experiment_end')
    memory_summary = memory_tracker.get_summary()
    peak_memory = memory_tracker.get_peak_memory()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - experiment_start
    
    log_print("\n" + "=" * 80)
    log_print("FINAL RESULTS")
    log_print("=" * 80)
    log_print(f"\nValidation Metrics:")
    log_print(f"  Accuracy:  {val_metrics['accuracy']*100:.2f}%")
    log_print(f"  Precision: {val_metrics['precision']:.4f}")
    log_print(f"  Recall:    {val_metrics['recall']:.4f}")
    log_print(f"  F1-Score:  {val_metrics['f1_score']:.4f}")
    
    log_print(f"\nTest Metrics:")
    log_print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    log_print(f"  Precision: {test_metrics['precision']:.4f}")
    log_print(f"  Recall:    {test_metrics['recall']:.4f}")
    log_print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    if 'auroc' in test_metrics:
        log_print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    
    log_print(f"\nTotal Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # FLOPs Summary
    if FLOPS_AVAILABLE and 'error' not in flops_report:
        log_print(f"\nFLOPs Summary:")
        log_print(f"  Total GFLOPs:              {flops_report['total_gflops']:.2f}")
        log_print(f"  Single Model GFLOPs:       {flops_report['single_model_forward_gflops']:.4f}")
        log_print(f"  Number of Models:          {flops_report['n_sources']}")
        log_print(f"  Merging GFLOPs:            {flops_report['merging_gflops']:.4f}")
        log_print(f"  Training GFLOPs:           {flops_report['training_gflops']:.2f}")
        log_print(f"  Validation Inference:      {flops_report['val_inference_gflops']:.2f}")
        log_print(f"  Test Inference:            {flops_report['test_inference_gflops']:.2f}")
    
    # GPU Memory Usage
    if memory_tracker.is_cuda:
        log_print(f"\nGPU Memory Usage:")
        log_print(f"  Actual GPU Memory (nvidia-smi): {peak_memory['actual_peak_gpu_memory_mb']:.2f} MB ({peak_memory['actual_peak_gpu_memory_gb']:.4f} GB)")
        log_print(f"  PyTorch Allocated:              {peak_memory['pytorch_peak_allocated_mb']:.2f} MB ({peak_memory['pytorch_peak_allocated_gb']:.4f} GB)")
        log_print(f"  PyTorch Reserved:               {peak_memory['pytorch_peak_reserved_mb']:.2f} MB ({peak_memory['pytorch_peak_reserved_gb']:.4f} GB)")
    else:
        log_print(f"\nNo GPU used (CPU mode)")
    log_print("=" * 80)
    
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    save_confusion_matrix(val_labels, val_preds, cm_dir / "val_confusion_matrix.png")
    save_confusion_matrix(test_labels, test_preds, cm_dir / "test_confusion_matrix.png")
    
    create_training_curves(history, output_dir / "training_curves.png")
    
    results = {
        'dataset': dataset,
        'target': target,
        'sources': sources,
        'trial': trial_num,
        'model': args.model,
        'method': 'fusion',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'hyperparameters': {
            'finetune_epochs': args.finetune_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
        },
        'num_source_models': len(source_models),
        'gpu_memory': memory_tracker.get_memory_dict(),
        'memory_checkpoints': memory_summary['checkpoints'],
        'flops': flops_report if FLOPS_AVAILABLE else {'error': 'FLOPs counter not available'},
        'dataset_sizes': {
            'n_train_samples': n_train_samples,
            'n_val_samples': n_val_samples,
            'n_test_samples': n_test_samples
        }
    }
    
    # Save memory profile
    memory_file = output_dir / "memory_profile.json"
    memory_tracker.save_profile(memory_file)
    
    # Save FLOPs report separately
    if FLOPS_AVAILABLE and 'error' not in flops_report:
        flops_file = output_dir / "flops_report.json"
        with open(flops_file, 'w') as f:
            json.dump(flops_report, f, indent=2)
    
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_print(f"\nResults saved to: {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Fusion Transfer Learning Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['yearbook', 'clear10', 'clear100'],
                        help='Dataset name')
    parser.add_argument('--target', type=str, required=True,
                        help='Target subset')
    parser.add_argument('--sources', nargs='+', required=True,
                        help='Source subsets')
    parser.add_argument('--model', type=str, default='apple/mobilevit-xx-small',
                        help='Model architecture')
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    
    args = parser.parse_args()

    # Generate random seed if not provided
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Using randomly generated seed: {args.seed}")
    
    pl.seed_everything(args.seed, workers=True)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"FUSION TRANSFER LEARNING - {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"Target: {args.target}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"FLOPs Tracking: {'Enabled' if FLOPS_AVAILABLE else 'Disabled (module not found)'}")
    print(f"{'='*80}\n")
    
    results = run_fusion_experiment(
        dataset=args.dataset,
        target=args.target,
        sources=args.sources,
        args=args,
        device=device,
        trial_num=args.trial
    )
    
    if results:
        print(f"\nFusion experiment complete!")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")
        print(f"  Time: {results['total_time_minutes']:.2f} minutes")
        if FLOPS_AVAILABLE and 'error' not in results.get('flops', {}):
            print(f"  Total FLOPs: {results['flops']['total_gflops']:.2f} GFLOPs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
