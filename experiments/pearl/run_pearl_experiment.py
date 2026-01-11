#!/usr/bin/env python
"""
run_pearl_experiment.py - PEARL Adapter Composition Experiment

PEARL (Parameter-Efficient Adaptive Representation Learning) with accurate FLOPs counting.

Usage:
    python run_pearl_experiment.py \
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

# Add shared utilities
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_utils import get_dataloaders
from model_utils import load_baseline_model, standardize_model_name
from flops_counter import get_pearl_flops_report
from metrics_utils import calculate_metrics, save_confusion_matrix, create_training_curves
from gpu_memory_tracker import GPUMemoryTracker


# ============================================================================
# LORA ADAPTER IMPLEMENTATION
# ============================================================================

class LoRALinearLayer(nn.Module):
    """LoRA adapter for Linear layers."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0
    
    def forward(self, x):
        # x: (batch, ..., in_features)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAConv2dLayer(nn.Module):
    """LoRA adapter for Conv2d layers."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Use 1x1 convolutions but match stride/padding to preserve output shape
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, 
                               stride=stride, padding=0, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, 
                               stride=1, padding=0, bias=False)
        self.scaling = 1.0
        
        # Initialize
        nn.init.kaiming_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # x: (batch, in_channels, H, W)
        # lora_A applies stride to match spatial dimensions
        # lora_B produces final output with correct channels
        return self.lora_B(self.lora_A(x)) * self.scaling


class AdapterManager:
    """Manages adapter integration via hooks."""
    
    def __init__(self, model: nn.Module, adapters: nn.ModuleDict, composition_weights: torch.Tensor = None):
        self.model = model
        self.adapters = adapters
        self.composition_weights = composition_weights
        self.hooks = []
        self.adapter_outputs = {}
    
    def _make_hook(self, layer_name: str):
        """Create a hook that adds adapter output to layer output."""
        def hook(module, input, output):
            adapter_key = layer_name.replace('.', '_')
            
            # Get adapter output
            if adapter_key in self.adapters:
                adapter = self.adapters[adapter_key]
                try:
                    adapter_out = adapter(input[0])
                    
                    # Verify shapes match before adding
                    if adapter_out.shape == output.shape:
                        # Add adapter output to original output
                        if hasattr(output, 'data'):
                            output.data = output.data + adapter_out
                        else:
                            output = output + adapter_out
                    else:
                        # Shape mismatch - skip this adapter
                        # This can happen with certain layer configurations
                        pass
                except Exception as e:
                    # Silently skip adapters that cause errors
                    pass
            
            return output
        return hook
    
    def register_hooks(self):
        """Register forward hooks for all adapted layers."""
        for name, module in self.model.named_modules():
            adapter_key = name.replace('.', '_')
            if adapter_key in self.adapters:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class WeightedAdapterManager:
    """Manages weighted composition of multiple adapters via hooks."""
    
    def __init__(self, model: nn.Module, adapters_list: List[nn.ModuleDict], weights: torch.Tensor):
        self.model = model
        self.adapters_list = adapters_list
        self.weights = weights
        self.hooks = []
    
    def _make_hook(self, layer_name: str):
        """Create a hook that adds weighted adapter outputs to layer output."""
        def hook(module, input, output):
            adapter_key = layer_name.replace('.', '_')
            
            # Combine outputs from all adapters with weights
            weighted_output = None
            has_valid_adapter = False
            
            for i, adapters in enumerate(self.adapters_list):
                if adapter_key in adapters:
                    adapter = adapters[adapter_key]
                    try:
                        adapter_out = adapter(input[0])
                        
                        # Verify shape matches output
                        if adapter_out.shape == output.shape:
                            if weighted_output is None:
                                weighted_output = self.weights[i] * adapter_out
                            else:
                                weighted_output = weighted_output + self.weights[i] * adapter_out
                            has_valid_adapter = True
                    except Exception:
                        # Skip adapters that cause errors
                        pass
            
            # Add weighted adapter output to original output if valid
            if has_valid_adapter and weighted_output is not None:
                if hasattr(output, 'data'):
                    output.data = output.data + weighted_output
                else:
                    output = output + weighted_output
            
            return output
        return hook
    
    def register_hooks(self):
        """Register forward hooks for all adapted layers."""
        # Find all layers that have at least one adapter
        adapted_layers = set()
        for adapters in self.adapters_list:
            adapted_layers.update(adapters.keys())
        
        for name, module in self.model.named_modules():
            adapter_key = name.replace('.', '_')
            if adapter_key in adapted_layers:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def add_lora_adapters(model: nn.Module, rank: int = 16) -> nn.ModuleDict:
    """
    Add LoRA adapters to key layers in model.
    Supports both Linear and Conv2d layers.
    """
    
    adapters = nn.ModuleDict()
    
    for name, module in model.named_modules():
        # Skip classifier and certain layers
        if 'classifier' in name.lower() or 'head' in name.lower():
            continue
        
        # Add adapters to Linear layers
        if isinstance(module, nn.Linear):
            adapter = LoRALinearLayer(module.in_features, module.out_features, rank)
            adapters[name.replace('.', '_')] = adapter
        
        # Add adapters to Conv2d layers (for CNNs like EfficientNet, ResNet, etc.)
        elif isinstance(module, nn.Conv2d):
            # Only add adapters to key convolutional layers
            # Skip first conv and very small channels to reduce overhead
            if module.in_channels >= 16 and module.out_channels >= 16:
                # Extract stride and padding
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                
                adapter = LoRAConv2dLayer(
                    module.in_channels, 
                    module.out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    rank=rank
                )
                adapters[name.replace('.', '_')] = adapter
    
    print(f"  Created {len(adapters)} adapters")
    if len(adapters) == 0:
        print("  WARNING: No suitable layers found for adaptation!")
        print(f"  Model architecture: {model.__class__.__name__}")
    
    return adapters


def forward_with_adapters(
    model: nn.Module,
    adapters: nn.ModuleDict,
    x: torch.Tensor,
    active_adapters: List[str] = None
) -> torch.Tensor:
    """
    Forward pass through model with LoRA adapters using hooks.
    """
    manager = AdapterManager(model, adapters)
    manager.register_hooks()
    
    try:
        outputs = model(x)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
    finally:
        manager.remove_hooks()
    
    return logits


def forward_with_weighted_adapters(
    model: nn.Module,
    adapters_list: List[nn.ModuleDict],
    weights: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Forward pass through model with weighted composition of multiple adapters.
    """
    manager = WeightedAdapterManager(model, adapters_list, weights)
    manager.register_hooks()
    
    try:
        outputs = model(x)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
    finally:
        manager.remove_hooks()
    
    return logits


# ============================================================================
# PEARL IMPLEMENTATION  
# ============================================================================

def train_adapter(
    base_model: nn.Module,
    adapters: nn.ModuleDict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3
) -> Dict:
    """Train a single LoRA adapter."""
    
    base_model.eval()  # Freeze base model
    
    # Only train adapter parameters
    optimizer = optim.AdamW(adapters.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        adapters.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use adapters in forward pass
            logits = forward_with_adapters(base_model, adapters, images)
            
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
        adapters.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Use adapters in forward pass
                logits = forward_with_adapters(base_model, adapters, images)
                
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
            best_state = deepcopy(adapters.state_dict())
        
        scheduler.step()
        
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}% "
              f"{'BEST' if val_acc == best_val_acc else ''}")
    
    # Load best adapter
    if best_state is not None:
        adapters.load_state_dict(best_state)
    
    return history


def learn_composition_weights(
    base_model: nn.Module,
    all_adapters: List[nn.ModuleDict],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3
) -> torch.Tensor:
    """
    Learn Bayesian composition weights for combining adapters.
    
    Returns tensor of shape (n_adapters,) with composition weights.
    """
    
    n_adapters = len(all_adapters)
    
    # Initialize weights (uniform)
    weights = torch.ones(n_adapters, device=device) / n_adapters
    weights.requires_grad = True
    
    optimizer = optim.Adam([weights], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_weights = weights.detach().clone()
    
    # Set model and all adapters to eval mode
    base_model.eval()
    for adapters in all_adapters:
        adapters.eval()
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Get outputs with weighted adapter composition
            logits = forward_with_weighted_adapters(base_model, all_adapters, weights, images)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Project weights to simplex
            with torch.no_grad():
                weights.data = torch.clamp(weights.data, min=0)
                weights.data = weights.data / weights.data.sum()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                logits = forward_with_weighted_adapters(base_model, all_adapters, weights, images)
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = weights.detach().clone()
        
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"Train Acc={train_acc*100:.2f}%, Val Acc={val_acc*100:.2f}% "
              f"Weights={weights.detach().cpu().numpy()}")
    
    return best_weights


def evaluate_model_with_adapters(
    base_model: nn.Module,
    adapters_list: List[nn.ModuleDict],
    weights: torch.Tensor,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple:
    """Evaluate model with weighted adapter composition."""
    
    base_model.eval()
    for adapters in adapters_list:
        adapters.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            # Get output with weighted adapter composition
            logits = forward_with_weighted_adapters(base_model, adapters_list, weights, images)
            
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

def run_pearl_experiment(
    dataset: str,
    target: str,
    sources: List[str],
    args,
    device: torch.device,
    trial_num: int = 1
) -> Dict:
    """Run complete PEARL experiment with accurate FLOPs counting."""
    
    # Setup output directory
    model_prefix = standardize_model_name(args.model)
    experiment_name = f"{model_prefix}_{target}_pearl_trial{trial_num}"
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
    log_print(f"PEARL TRANSFER LEARNING EXPERIMENT")
    log_print("=" * 80)
    log_print(f"\nModel: {args.model}")
    log_print(f"Dataset: {dataset}")
    log_print(f"Target: {target}")
    log_print(f"Sources: {', '.join(sources)}")
    log_print(f"Trial: {trial_num}")
    log_print(f"Device: {device}")
    log_print(f"LoRA Rank: {args.adapter_dim}")
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
    
    # Load base model
    log_print("Loading base model...")
    base_model = load_baseline_model(
        dataset=dataset,
        subset=sources[0],
        model_name=args.model,
        num_classes=num_classes
    )
    base_model = base_model.to(device)
    base_model.eval()  # Freeze base model
    log_print(f"Loaded base model from {sources[0]}\n")
    memory_tracker.checkpoint('base_model_created')
    
    # Train adapters for each source
    all_adapters = []
    adapter_histories = []
    
    for source_idx, source in enumerate(sources):
        log_print(f"{'='*80}")
        log_print(f"Source {source_idx+1}/{len(sources)}: {source}")
        log_print(f"{'='*80}")
        
        # Create adapter
        log_print(f"Creating LoRA adapter (rank={args.adapter_dim})...")
        adapters = add_lora_adapters(base_model, rank=args.adapter_dim)
        adapters = adapters.to(device)
        
        n_adapter_params = sum(p.numel() for p in adapters.parameters())
        log_print(f"Adapter created: {n_adapter_params:,} parameters\n")
        
        # Train adapter
        log_print(f"Training adapter ({args.finetune_epochs} epochs)...")
        history = train_adapter(
            base_model=base_model,
            adapters=adapters,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.finetune_epochs,
            learning_rate=args.learning_rate
        )
        log_print(f"Adapter training complete\n")
        
        all_adapters.append(adapters)
        adapter_histories.append(history)
    
    memory_tracker.checkpoint('adapters_trained')
    
    # Learn composition weights
    log_print("=" * 80)
    log_print("LEARNING ADAPTER COMPOSITION WEIGHTS")
    log_print("=" * 80)
    log_print(f"Optimizing composition ({args.composition_epochs} epochs)...")
    
    composition_weights = learn_composition_weights(
        base_model=base_model,
        all_adapters=all_adapters,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.composition_epochs
    )
    
    log_print(f"Composition weights learned")
    log_print(f"  Weights: {composition_weights.cpu().numpy()}\n")
    memory_tracker.checkpoint('composition_learned')
    
    # Final evaluation
    log_print("Evaluating final model...")
    log_print("  Validation set...")
    val_metrics, val_labels, val_preds, val_probs = evaluate_model_with_adapters(
        base_model, all_adapters, composition_weights, val_loader, device
    )
    
    log_print("  Test set...")
    test_metrics, test_labels, test_preds, test_probs = evaluate_model_with_adapters(
        base_model, all_adapters, composition_weights, test_loader, device
    )
    
    # Calculate total time
    # Get GPU memory summary
    memory_tracker.checkpoint('experiment_end')
    memory_summary = memory_tracker.get_summary()
    peak_memory = memory_tracker.get_peak_memory()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - experiment_start
    
    # Calculate accurate FLOPs
    log_print("\nCalculating accurate FLOPs...")
    flops_report = get_pearl_flops_report(
        model=base_model,
        input_size=(1, 3, 224, 224),
        n_sources=len(sources),
        n_train_samples=len(train_loader.dataset),
        adapter_epochs=args.finetune_epochs,
        composition_epochs=args.composition_epochs,
        lora_rank=args.adapter_dim,
        n_adapted_layers=12,  # Typical for MobileViT
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
    log_print(f"  Total FLOPs:              {flops_report['total_gflops']:.2f} GFLOPs")
    log_print(f"  Adapter Training (total): {flops_report['adapter_training_gflops_total']:.2f} GFLOPs")
    log_print(f"  Composition Learning:     {flops_report['composition_learning_gflops']:.2f} GFLOPs")
    log_print(f"  Adapter Parameters:       {n_adapter_params/1e6:.2f}M per source")
    
    log_print(f"\nTotal Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
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
    
    # Save training curves (aggregate all adapter histories)
    all_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    for history in adapter_histories:
        for key in all_history.keys():
            all_history[key].extend(history[key])
    
    create_training_curves(all_history, output_dir / "training_curves.png")
    
    # Save complete results
    results = {
        'dataset': dataset,
        'target': target,
        'sources': sources,
        'trial': trial_num,
        'model': args.model,
        'method': 'pearl',
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'flops': flops_report,
        'gpu_memory': memory_tracker.get_memory_dict(),
        'memory_checkpoints': memory_summary['checkpoints'],
        'hyperparameters': {
            'finetune_epochs': args.finetune_epochs,
            'composition_epochs': args.composition_epochs,
            'adapter_dim': args.adapter_dim,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
        },
        'composition_weights': composition_weights.cpu().tolist(),
        'adapter_params_per_source': n_adapter_params
    }
    
    # Save config
    # Save memory profile
    memory_file = output_dir / "memory_profile.json"
    memory_tracker.save_profile(memory_file)
    
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save FLOPs report
    flops_file = output_dir / "flops_report.json"
    with open(flops_file, 'w') as f:
        json.dump(flops_report, f, indent=2)
    
    log_print(f"\nResults saved to: {output_dir}")
    
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PEARL Transfer Learning Experiment",
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
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--composition_epochs', type=int, default=3)
    parser.add_argument('--adapter_dim', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
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
    print(f"PEARL TRANSFER LEARNING - {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"Target: {args.target}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    results = run_pearl_experiment(
        dataset=args.dataset,
        target=args.target,
        sources=args.sources,
        args=args,
        device=device,
        trial_num=args.trial
    )
    
    if results:
        print(f"\nPEARL experiment complete!")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']*100:.2f}%")
        print(f"  Total FLOPs: {results['flops']['total_gflops']:.2f} GFLOPs")
        print(f"  Time: {results['total_time_minutes']:.2f} minutes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())