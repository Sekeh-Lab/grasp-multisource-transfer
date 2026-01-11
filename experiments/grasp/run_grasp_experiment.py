#!/usr/bin/env python
"""
run_grasp_experiment.py - GRASP Sequential Transfer Experiment

GRASP (Gradient-Aligned Sequential Parameter Transfer) with accurate FLOPs counting.

Usage:
    python run_grasp_experiment.py \
        --dataset clear10 \
        --target year_1-2 \
        --sources year_3-4 year_5-6 year_7-8 year_9-10 \
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add shared utilities
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_utils import get_dataloaders
from model_utils import load_baseline_model, standardize_model_name
from flops_counter import get_grasp_flops_report
from metrics_utils import calculate_metrics, save_confusion_matrix, create_training_curves
from gpu_memory_tracker import GPUMemoryTracker


# ============================================================================
# GRASP IMPLEMENTATION
# ============================================================================

def compute_gradient_alignment(
    source_model: nn.Module,
    target_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Compute gradient alignment between source and target models.
    
    Returns dict mapping parameter names to alignment scores.
    """
    source_model.eval()
    target_model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    # Accumulate gradients
    source_grads = {}
    target_grads = {}
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        # Source gradients
        source_model.zero_grad()
        outputs = source_model(images)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        loss = criterion(logits, labels)
        loss.backward()
        
        for name, param in source_model.named_parameters():
            if param.grad is not None:
                if name not in source_grads:
                    source_grads[name] = param.grad.detach().clone()
                else:
                    source_grads[name] += param.grad.detach().clone()
        
        # Target gradients
        target_model.zero_grad()
        outputs = target_model(images)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        loss = criterion(logits, labels)
        loss.backward()
        
        for name, param in target_model.named_parameters():
            if param.grad is not None:
                if name not in target_grads:
                    target_grads[name] = param.grad.detach().clone()
                else:
                    target_grads[name] += param.grad.detach().clone()
    
    # Compute alignment scores (cosine similarity)
    alignment_scores = {}
    for name in source_grads.keys():
        if name in target_grads:
            src_grad = source_grads[name].flatten()
            tgt_grad = target_grads[name].flatten()
            
            # Cosine similarity
            similarity = torch.dot(src_grad, tgt_grad) / (
                torch.norm(src_grad) * torch.norm(tgt_grad) + 1e-8
            )
            alignment_scores[name] = similarity.item()
    
    return alignment_scores


def transfer_aligned_parameters(
    source_model: nn.Module,
    target_model: nn.Module,
    alignment_scores: Dict[str, float],
    threshold: float = 0.0
) -> int:
    """
    Transfer parameters from source to target based on alignment scores.
    
    Returns number of parameters transferred.
    """
    n_transferred = 0
    
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    
    for name, score in alignment_scores.items():
        if score > threshold and name in source_state and name in target_state:
            target_state[name] = source_state[name].clone()
            n_transferred += source_state[name].numel()
    
    target_model.load_state_dict(target_state)
    
    return n_transferred


def fine_tune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 5e-5
) -> Dict:
    """Fine-tune model on target data."""
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
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
            best_state = deepcopy(model.state_dict())
        
        scheduler.step()
        
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}% "
              f"{'BEST' if val_acc == best_val_acc else ''}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple:
    """Evaluate model and return metrics."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
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


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_grasp_experiment(
    dataset: str,
    target: str,
    sources: List[str],
    args,
    device: torch.device,
    trial_num: int = 1
) -> Dict:
    """Run complete GRASP experiment with accurate FLOPs counting."""
    
    # Setup output directory with threshold in name
    model_prefix = standardize_model_name(args.model)
    threshold_str = str(int(args.alignment_threshold * 100))
    experiment_name = f"{model_prefix}_{target}_grasp_{threshold_str}_trial{trial_num}"
    output_dir = Path(args.results_dir) / dataset / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{experiment_name}_comprehensive_report.txt"
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + "\n")
    
    # Start experiment
    experiment_start = time.perf_counter()
    
    # Initialize GPU memory tracker
    memory_tracker = GPUMemoryTracker(device)
    memory_tracker.checkpoint('experiment_start')
    
    log_print("=" * 80)
    log_print(f"GRASP TRANSFER LEARNING EXPERIMENT")
    log_print("=" * 80)
    log_print(f"\nModel: {args.model}")
    log_print(f"Dataset: {dataset}")
    log_print(f"Target: {target}")
    log_print(f"Sources: {', '.join(sources)}")
    log_print(f"Trial: {trial_num}")
    log_print(f"Alignment Threshold: {args.alignment_threshold}")
    log_print(f"Device: {device}")
    log_print("=" * 80 + "\n")
    
    # Load data
    log_print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset=dataset,
        target=target,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    log_print(f"Data loaded: {num_classes} classes")
    log_print(f"  Train samples: {len(train_loader.dataset)}")
    log_print(f"  Val samples: {len(val_loader.dataset)}")
    log_print(f"  Test samples: {len(test_loader.dataset)}\n")
    memory_tracker.checkpoint('data_loaded')
    
    # Load initial model
    log_print("Loading initial model...")
    current_model = load_baseline_model(
        dataset=dataset,
        subset=sources[0],
        model_name=args.model,
        num_classes=num_classes
    )
    current_model = current_model.to(device)
    log_print(f"Loaded initial model from {sources[0]}\n")
    memory_tracker.checkpoint('initial_model_loaded')
    
    # Fine-tune initial model for initial_epochs
    log_print(f"{'='*80}")
    log_print(f"Initial Fine-tuning on Target Data")
    log_print(f"{'='*80}")
    log_print(f"Fine-tuning initial model for {args.initial_epochs} epochs...")
    initial_history = fine_tune_model(
        model=current_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.initial_epochs,
        learning_rate=args.learning_rate
    )
    log_print(f"Initial fine-tuning complete\n")
    memory_tracker.checkpoint('initial_finetuning_complete')
    
    # Sequential transfer from remaining sources
    transfer_history = []
    
    for source_idx, source in enumerate(sources[1:], start=1):
        log_print(f"{'='*80}")
        log_print(f"Source {source_idx}/{len(sources)-1}: {source}")
        log_print(f"{'='*80}")
        
        # Load source model
        log_print(f"Loading source model from {source}...")
        source_model = load_baseline_model(
            dataset=dataset,
            subset=source,
            model_name=args.model,
            num_classes=num_classes
        )
        source_model = source_model.to(device)
        log_print(f"Loaded source model\n")
        
        # Compute gradient alignment
        log_print(f"Computing gradient alignment ({args.num_alignment_batches} batches)...")
        alignment_scores = compute_gradient_alignment(
            source_model=source_model,
            target_model=current_model,
            data_loader=train_loader,
            device=device,
            num_batches=args.num_alignment_batches
        )
        log_print(f"Computed alignment for {len(alignment_scores)} parameters\n")
        
        # Transfer parameters
        log_print(f"Transferring aligned parameters (threshold={args.alignment_threshold})...")
        n_transferred = transfer_aligned_parameters(
            source_model=source_model,
            target_model=current_model,
            alignment_scores=alignment_scores,
            threshold=args.alignment_threshold
        )
        log_print(f"Transferred {n_transferred:,} parameters\n")
        
        # Fine-tune
        log_print(f"Fine-tuning ({args.finetune_epochs} epochs)...")
        history = fine_tune_model(
            model=current_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.finetune_epochs,
            learning_rate=args.learning_rate
        )
        log_print(f"Fine-tuning complete\n")
        
        transfer_history.append({
            'source': source,
            'n_transferred': n_transferred,
            'history': history
        })
    
    # Final evaluation
    log_print("Evaluating final model...")
    log_print("  Validation set...")
    val_metrics, val_labels, val_preds, val_probs = evaluate_model(
        current_model, val_loader, device
    )
    
    log_print("  Test set...")
    test_metrics, test_labels, test_preds, test_probs = evaluate_model(
        current_model, test_loader, device
    )
    
    # Calculate total time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - experiment_start
    
    # Get GPU memory summary
    memory_tracker.checkpoint('experiment_end')
    memory_summary = memory_tracker.get_summary()
    peak_memory = memory_tracker.get_peak_memory()
    
        # Calculate accurate FLOPs
    log_print("\nCalculating accurate FLOPs...")
    flops_report = get_grasp_flops_report(
        model=current_model,
        input_size=(1, 3, 224, 224),
        n_sources=len(sources),
        n_train_samples=len(train_loader.dataset),
        initial_epochs=args.initial_epochs,
        finetune_epochs=args.finetune_epochs,
        alignment_batches=args.num_alignment_batches,
        batch_size=args.batch_size,
        device=device
    )
    log_print(f"FLOPs calculated: {flops_report['total_gflops']:.2f} GFLOPs total\n")
    
    # Save results
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
    
    log_print(f"\nComputational Cost:")
    log_print(f"  Total FLOPs:           {flops_report['total_gflops']:.2f} GFLOPs")
    log_print(f"  Initial Training:      {flops_report['initial_training_gflops']:.2f} GFLOPs")
    log_print(f"  Alignment (per src):   {flops_report['alignment_gflops_total']/max(1,len(sources)-1):.4f} GFLOPs")
    log_print(f"  Transfer (per src):    {flops_report['transfer_gflops_total']/max(1,len(sources)-1):.4f} GFLOPs")
    log_print(f"  Fine-tuning (total):   {flops_report['finetune_gflops_total']:.2f} GFLOPs")
    
    log_print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # GPU Memory Usage
    if memory_tracker.is_cuda:
        log_print(f"\nGPU Memory Usage:")
        log_print(f"  Actual GPU Memory (nvidia-smi): {peak_memory['actual_peak_gpu_memory_mb']:.2f} MB ({peak_memory['actual_peak_gpu_memory_gb']:.4f} GB)")
        log_print(f"  PyTorch Allocated:              {peak_memory['pytorch_peak_allocated_mb']:.2f} MB ({peak_memory['pytorch_peak_allocated_gb']:.4f} GB)")
        log_print(f"  PyTorch Reserved:               {peak_memory['pytorch_peak_reserved_mb']:.2f} MB ({peak_memory['pytorch_peak_reserved_gb']:.4f} GB)")
    else:
        log_print(f"\nNo GPU used (CPU mode)")
    log_print("=" * 80)
    
    # Save confusion matrices
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    save_confusion_matrix(val_labels, val_preds, cm_dir / "val_confusion_matrix.png")
    save_confusion_matrix(test_labels, test_preds, cm_dir / "test_confusion_matrix.png")
    
    # Save training curves (aggregate initial history and all transfer histories)
    all_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Add initial training history
    for key in all_history.keys():
        all_history[key].extend(initial_history[key])
    
    # Add transfer histories
    for th in transfer_history:
        for key in all_history.keys():
            all_history[key].extend(th['history'][key])
    
    create_training_curves(all_history, output_dir / "training_curves.png")
    
    # Save complete results
    results = {
        'dataset': dataset,
        'target': target,
        'sources': sources,
        'trial': trial_num,
        'model': args.model,
        'method': 'grasp',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'flops': flops_report,
        'gpu_memory': memory_tracker.get_memory_dict(),
        'memory_checkpoints': memory_summary['checkpoints'],
        'hyperparameters': {
            'initial_epochs': args.initial_epochs,
            'finetune_epochs': args.finetune_epochs,
            'num_alignment_batches': args.num_alignment_batches,
            'alignment_threshold': args.alignment_threshold,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
        },
        'transfer_history': [{
            'source': th['source'],
            'n_transferred': th['n_transferred']
        } for th in transfer_history]
    }
    
    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save memory profile
    memory_file = output_dir / "memory_profile.json"
    memory_tracker.save_profile(memory_file)
    
    # Save FLOPs report
    flops_file = output_dir / "flops_report.json"
    with open(flops_file, 'w') as f:
        json.dump(flops_report, f, indent=2)
    
    # Save model
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(current_model.state_dict(), 
               checkpoint_dir / f"{model_prefix}_{target}_grasp_{threshold_str}_final.ckpt")
    
    log_print(f"\nResults saved to: {output_dir}")
    
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GRASP Transfer Learning Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['yearbook', 'clear10', 'clear100'])
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--model', type=str, default='apple/mobilevit-xx-small',
                        help='Model name (any format accepted by model_utils)')
    parser.add_argument('--trial', type=int, default=1)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--initial_epochs', type=int, default=3)
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--num_alignment_batches', type=int, default=1)
    parser.add_argument('--alignment_threshold', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    
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
    print(f"GRASP TRANSFER LEARNING - {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"Target: {args.target}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Model: {args.model}")
    print(f"Alignment Threshold: {args.alignment_threshold}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    results = run_grasp_experiment(
        dataset=args.dataset,
        target=args.target,
        sources=args.sources,
        args=args,
        device=device,
        trial_num=args.trial
    )
    
    if results:
        print(f"\nGRASP experiment complete!")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")
        print(f"  Total FLOPs: {results['flops']['total_gflops']:.2f} GFLOPs")
        print(f"  Time: {results['total_time_minutes']:.2f} minutes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())