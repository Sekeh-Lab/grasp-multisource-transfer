#!/usr/bin/env python
"""
run_ensemble_experiment.py - Unified Ensemble Experiment with FLOPs Tracking

Voting ensemble: Load multiple source models and combine predictions.

Usage:
    python run_ensemble_experiment.py \
        --dataset clear100 \
        --target year_1-2 \
        --sources year_3-4 year_5-6 year_7-8 year_9-10 \
        --trial 1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_utils import get_dataloaders
from model_utils import load_baseline_model, standardize_model_name
from metrics_utils import calculate_metrics, save_confusion_matrix
from gpu_memory_tracker import GPUMemoryTracker

# Import FLOPs counter
try:
    from flops_counter import count_ensemble_total_flops, get_ensemble_flops_report
    FLOPS_AVAILABLE = True
except ImportError:
    print("Warning: flops_counter not available. FLOPs will not be calculated.")
    FLOPS_AVAILABLE = False


class VotingEnsemble(nn.Module):
    """
    Voting ensemble that combines predictions from multiple models.
    Supports both hard voting (majority vote) and soft voting (average probabilities).
    """
    
    def __init__(self, models: List[nn.Module], voting: str = 'soft'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting
        weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, x):
        """Forward pass through all models and combine predictions."""
        logits_list = []
        
        for model in self.models:
            with torch.no_grad():
                outputs = model(x)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict):
                    logits = outputs.get('logits', list(outputs.values())[0])
                else:
                    logits = outputs
                logits_list.append(logits)
        
        stacked_logits = torch.stack(logits_list)
        
        if self.voting == 'soft':
            avg_logits = (stacked_logits * self.weights.view(-1, 1, 1)).sum(dim=0)
            return avg_logits
        else:
            probs = torch.softmax(stacked_logits, dim=-1)
            avg_probs = (probs * self.weights.view(-1, 1, 1)).sum(dim=0)
            return torch.log(avg_probs + 1e-10)


def evaluate_ensemble(
    ensemble: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict:
    """Evaluate ensemble model."""
    
    ensemble.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            
            logits = ensemble(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    if num_classes == 2:
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            metrics['auroc'] = auroc
        except Exception as e:
            print(f"Warning: Could not calculate AUROC: {e}")
    elif num_classes > 2:
        try:
            unique_classes = np.unique(all_labels)
            if len(unique_classes) == num_classes and all_probs.shape[1] == num_classes:
                auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                metrics['auroc'] = auroc
            else:
                print(f"Warning: AUROC skipped - classes mismatch (labels: {len(unique_classes)}, probs: {all_probs.shape[1]})")
        except Exception as e:
            print(f"Warning: Could not calculate AUROC: {e}")
    
    return metrics, all_labels, all_preds, all_probs


def calculate_flops_for_ensemble(
    single_model: nn.Module,
    n_sources: int,
    n_val_samples: int,
    n_test_samples: int,
    num_classes: int,
    voting: str,
    device: torch.device,
    input_size: tuple = (1, 3, 224, 224)
) -> Dict:
    """Calculate FLOPs for ensemble experiment."""
    
    if not FLOPS_AVAILABLE:
        return {
            'method': 'ENSEMBLE',
            'error': 'FLOPs counter not available',
            'total_gflops': 0.0
        }
    
    try:
        flops_report = get_ensemble_flops_report(
            model=single_model,
            input_size=input_size,
            n_sources=n_sources,
            n_val_samples=n_val_samples,
            n_test_samples=n_test_samples,
            num_classes=num_classes,
            voting=voting,
            device=device
        )
        return flops_report
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return {
            'method': 'ENSEMBLE',
            'error': str(e),
            'total_gflops': 0.0
        }


def run_ensemble_experiment(
    dataset: str,
    target: str,
    sources: List[str],
    args,
    device: torch.device,
    trial_num: int = 1
) -> Dict:
    """Run complete ensemble experiment."""
    
    model_prefix = standardize_model_name(args.model)
    experiment_name = f"{model_prefix}_{target}_ensemble_trial{trial_num}"
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
    log_print("VOTING ENSEMBLE EXPERIMENT")
    log_print("=" * 80)
    log_print(f"Model: {args.model}")
    log_print(f"Dataset: {dataset}")
    log_print(f"Target: {target}")
    log_print(f"Sources: {', '.join(sources)}")
    log_print(f"Trial: {trial_num}")
    log_print(f"Device: {device}")
    log_print(f"Method: {args.voting.capitalize()} Voting")
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
    n_val_samples = len(val_loader.dataset)
    n_test_samples = len(test_loader.dataset)
    
    log_print(f"Data loaded: {num_classes} classes")
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
        log_print("\nERROR: No source models loaded. Cannot create ensemble.")
        return None
    
    log_print(f"Loaded {len(source_models)}/{len(sources)} source models\n")
    memory_tracker.checkpoint('source_models_loaded')
    
    log_print("Creating voting ensemble...")
    ensemble = VotingEnsemble(source_models, voting=args.voting)
    ensemble = ensemble.to(device)
    ensemble.eval()
    log_print(f"Ensemble created with {args.voting} voting\n")
    memory_tracker.checkpoint('ensemble_created')
    
    # Calculate FLOPs
    log_print("Calculating FLOPs...")
    flops_report = calculate_flops_for_ensemble(
        single_model=source_models[0],
        n_sources=len(source_models),
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        num_classes=num_classes,
        voting=args.voting,
        device=device,
        input_size=(1, 3, 224, 224)
    )
    
    if FLOPS_AVAILABLE and 'error' not in flops_report:
        log_print(f"FLOPs calculated successfully")
        log_print(f"  Total GFLOPs: {flops_report['total_gflops']:.2f}")
        log_print(f"  Validation inference GFLOPs: {flops_report['val_inference_gflops']:.2f}")
        log_print(f"  Test inference GFLOPs: {flops_report['test_inference_gflops']:.2f}")
        log_print(f"  Combination GFLOPs: {flops_report['combination_gflops']:.4f}")
    else:
        log_print(f"FLOPs calculation skipped or failed")
    log_print("")
    
    log_print("Evaluating on validation set...")
    val_metrics, val_labels, val_preds, val_probs = evaluate_ensemble(
        ensemble, val_loader, device, num_classes
    )
    
    log_print("Evaluating on test set...")
    test_metrics, test_labels, test_preds, test_probs = evaluate_ensemble(
        ensemble, test_loader, device, num_classes
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
    if 'auroc' in val_metrics:
        log_print(f"  AUROC:     {val_metrics['auroc']:.4f}")
    
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
        log_print(f"  Single Model GFLOPs:       {flops_report['single_model_inference_gflops']:.4f}")
        log_print(f"  Number of Models:          {flops_report['n_sources']}")
        log_print(f"  Validation Inference:      {flops_report['val_inference_gflops']:.2f} GFLOPs")
        log_print(f"  Test Inference:            {flops_report['test_inference_gflops']:.2f} GFLOPs")
        log_print(f"  Combination Overhead:      {flops_report['combination_gflops']:.4f} GFLOPs")
        if 'softmax_gflops' in flops_report and flops_report['softmax_gflops'] > 0:
            log_print(f"  Softmax (Soft Voting):     {flops_report['softmax_gflops']:.4f} GFLOPs")
    
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
    
    results = {
        'dataset': dataset,
        'target': target,
        'sources': sources,
        'trial': trial_num,
        'model': args.model,
        'method': 'ensemble',
        'voting': args.voting,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'num_source_models': len(source_models),
        'gpu_memory': memory_tracker.get_memory_dict(),
        'memory_checkpoints': memory_summary['checkpoints'],
        'flops': flops_report if FLOPS_AVAILABLE else {'error': 'FLOPs counter not available'},
        'dataset_sizes': {
            'n_val_samples': n_val_samples,
            'n_test_samples': n_test_samples
        }
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
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
    
    log_print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Voting Ensemble Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['yearbook', 'clear10', 'clear100'])
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--model', type=str, default='apple/mobilevit-xx-small')
    parser.add_argument('--trial', type=int, default=1)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--voting', type=str, default='soft', choices=['soft', 'hard'])
    
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
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
    print(f"VOTING ENSEMBLE - {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"Target: {args.target}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Method: {args.voting.capitalize()} Voting (Average Predictions)")
    print(f"FLOPs Tracking: {'Enabled' if FLOPS_AVAILABLE else 'Disabled (module not found)'}")
    print(f"{'='*80}\n")
    
    results = run_ensemble_experiment(
        dataset=args.dataset,
        target=args.target,
        sources=args.sources,
        args=args,
        device=device,
        trial_num=args.trial
    )
    
    if results:
        print(f"\nEnsemble experiment complete!")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")
        print(f"  Time: {results['total_time_minutes']:.2f} minutes")
        if FLOPS_AVAILABLE and 'error' not in results.get('flops', {}):
            print(f"  Total FLOPs: {results['flops']['total_gflops']:.2f} GFLOPs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
