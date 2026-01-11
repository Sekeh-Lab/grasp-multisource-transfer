"""
Comprehensive FLOPs Calculation for All Sequential Transfer Learning Methods

This module provides precise FLOPs calculations for all transfer learning methods:
- ENSEMBLE: Voting ensemble with multiple model inference
- FUSION: Parameter averaging followed by fine-tuning
- GRASP: Gradient alignment with selective transfer
- PEARL: LoRA adapter training and composition
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List


# ============================================================================
# Base Model FLOPs Counting
# ============================================================================

def count_model_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device = None
) -> int:
    """
    Count FLOPs for a single forward pass through model.
    
    Uses hook-based counting for common layer types.
    """
    if device is None:
        device = next(model.parameters()).device
    
    total_flops = 0
    
    def count_flops_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            # Conv2D: 2 * C_in * C_out * K_h * K_w * H_out * W_out
            batch_size = output.shape[0]
            out_channels = module.out_channels
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_size = output.shape[2] * output.shape[3]
            flops = batch_size * out_channels * kernel_ops * output_size
            total_flops += flops
        
        elif isinstance(module, nn.Linear):
            # Linear: 2 * in_features * out_features * batch_size
            batch_size = output.shape[0]
            flops = 2 * module.in_features * module.out_features * batch_size
            total_flops += flops
        
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
            # Normalization: ~2 * num_elements (mean + std)
            flops = 2 * output.numel()
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(count_flops_hook))
    
    # Run forward pass
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Count FLOPs for a single forward pass.
    """
    return count_model_flops(model, input_tensor.shape, input_tensor.device)


def count_merging_flops(n_params: int, n_sources: int, ops_per_param: int = 2) -> int:
    """
    Count FLOPs for simple parameter merging.
    """
    return n_params * n_sources * ops_per_param


def count_training_flops(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_samples: int,
    n_epochs: int
) -> int:
    """
    Count FLOPs for training.
    
    Training = 3x forward pass FLOPs (forward + backward + optimizer)
    """
    single_forward = count_flops(model, input_tensor)
    return 3 * single_forward * n_samples * n_epochs


def count_inference_flops(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_samples: int
) -> int:
    """
    Count FLOPs for inference.
    """
    single_forward = count_flops(model, input_tensor)
    return single_forward * n_samples


# ============================================================================
# ENSEMBLE FLOPs Calculation
# ============================================================================

def count_ensemble_total_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_val_samples: int,
    n_test_samples: int,
    num_classes: int = 10,
    voting: str = 'soft',
    device: torch.device = None,
    **kwargs
) -> Dict:
    """
    Calculate total FLOPs for Ensemble method.
    
    Ensemble: Multiple models each perform inference, predictions are combined
    
    Phases:
    1. Validation inference: All N models perform forward passes
    2. Test inference: All N models perform forward passes
    3. Prediction combination: Weighted averaging of logits/probabilities
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_params = sum(p.numel() for p in model.parameters())
    forward_flops = count_model_flops(model, input_size, device)
    
    # Validation inference: N models perform forward passes
    val_inference_flops = n_sources * forward_flops * n_val_samples
    
    # Test inference: N models perform forward passes
    test_inference_flops = n_sources * forward_flops * n_test_samples
    
    # Combining predictions (weighted averaging of logits)
    # For each sample: sum N predictions, weighted average
    total_samples = n_val_samples + n_test_samples
    combination_ops_per_sample = n_sources * num_classes * 3  # multiply by weight, sum, divide
    combination_flops = total_samples * combination_ops_per_sample
    
    # Softmax computation (if soft voting)
    softmax_flops = 0
    if voting == 'soft':
        # Softmax per model per sample: exp(x) + sum + divide
        softmax_flops = n_sources * total_samples * num_classes * 4
    
    total_flops = (
        val_inference_flops +
        test_inference_flops +
        combination_flops +
        softmax_flops
    )
    
    return {
        'method': 'ENSEMBLE',
        'voting': voting,
        'model_params_millions': n_params / 1e6,
        'single_model_inference_gflops': forward_flops / 1e9,
        'val_inference_gflops': val_inference_flops / 1e9,
        'test_inference_gflops': test_inference_flops / 1e9,
        'combination_gflops': combination_flops / 1e9,
        'softmax_gflops': softmax_flops / 1e9,
        'total_gflops': total_flops / 1e9,
        'n_sources': n_sources,
        'n_val_samples': n_val_samples,
        'n_test_samples': n_test_samples,
        'breakdown': {
            'val_inference_percent': 100 * val_inference_flops / total_flops,
            'test_inference_percent': 100 * test_inference_flops / total_flops,
            'combination_percent': 100 * (combination_flops + softmax_flops) / total_flops,
        }
    }


def get_ensemble_flops_report(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_val_samples: int,
    n_test_samples: int,
    num_classes: int = 10,
    voting: str = 'soft',
    device: torch.device = None,
    **kwargs
) -> Dict:
    """Get Ensemble FLOPs report."""
    return count_ensemble_total_flops(
        model=model,
        input_size=input_size,
        n_sources=n_sources,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        num_classes=num_classes,
        voting=voting,
        device=device,
        **kwargs
    )


# ============================================================================
# FUSION FLOPs Calculation
# ============================================================================

def count_fusion_total_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    finetune_epochs: int = 3,
    device: torch.device = None,
    **kwargs
) -> Dict:
    """
    Calculate total FLOPs for Fusion method.
    
    Fusion: Merge multiple models via parameter averaging, then fine-tune
    
    Phases:
    1. Parameter fusion: Weighted averaging of N source model parameters
    2. Fine-tuning: Train fused model on target data
    3. Validation/test inference: Single fused model performs inference
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_params = sum(p.numel() for p in model.parameters())
    forward_flops = count_model_flops(model, input_size, device)
    
    # Phase 1: Parameter fusion
    # For each parameter: sum N source values, compute weighted average
    fusion_flops = n_params * n_sources * 2  # multiply by weight and sum
    
    # Phase 2: Fine-tuning the fused model
    # Training = forward + backward + optimizer update
    finetune_flops = 3 * forward_flops * n_train_samples * finetune_epochs
    
    # Phase 3: Inference (validation + test)
    # Only ONE fused model performs inference
    val_inference_flops = forward_flops * n_val_samples
    test_inference_flops = forward_flops * n_test_samples
    
    total_flops = (
        fusion_flops +
        finetune_flops +
        val_inference_flops +
        test_inference_flops
    )
    
    return {
        'method': 'FUSION',
        'model_params_millions': n_params / 1e6,
        'single_model_inference_gflops': forward_flops / 1e9,
        'fusion_gflops': fusion_flops / 1e9,
        'finetune_gflops': finetune_flops / 1e9,
        'val_inference_gflops': val_inference_flops / 1e9,
        'test_inference_gflops': test_inference_flops / 1e9,
        'total_gflops': total_flops / 1e9,
        'n_sources': n_sources,
        'finetune_epochs': finetune_epochs,
        'n_train_samples': n_train_samples,
        'n_val_samples': n_val_samples,
        'n_test_samples': n_test_samples,
        'breakdown': {
            'fusion_percent': 100 * fusion_flops / total_flops,
            'finetune_percent': 100 * finetune_flops / total_flops,
            'inference_percent': 100 * (val_inference_flops + test_inference_flops) / total_flops,
        }
    }


def get_fusion_flops_report(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    finetune_epochs: int = 3,
    device: torch.device = None,
    **kwargs
) -> Dict:
    """Get Fusion FLOPs report."""
    return count_fusion_total_flops(
        model=model,
        input_size=input_size,
        n_sources=n_sources,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_test_samples=n_test_samples,
        finetune_epochs=finetune_epochs,
        device=device,
        **kwargs
    )


# ============================================================================
# GRASP FLOPs Calculation
# ============================================================================

def count_grasp_total_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_train_samples: int,
    initial_epochs: int,
    finetune_epochs: int,
    alignment_batches: int = 1,
    batch_size: int = 32,
    device: torch.device = None,
    **kwargs
) -> Dict:
    """
    Calculate total FLOPs for GRASP method.
    
    GRASP: Gradient-Aligned Sequential Parameter Transfer
    
    Phases:
    1. Initial training on first source
    2. For each additional source:
       a. Gradient alignment (forward + backward on both models)
       b. Selective parameter transfer
       c. Fine-tuning
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_params = sum(p.numel() for p in model.parameters())
    forward_flops = count_model_flops(model, input_size, device)
    
    # Phase 1: Initial training on first source
    initial_training_flops = 3 * forward_flops * n_train_samples * initial_epochs
    
    # Phase 2: Sequential transfer from additional sources
    n_transfer_sources = n_sources - 1
    
    # Gradient alignment: forward + backward on BOTH models
    alignment_samples = alignment_batches * batch_size
    alignment_flops = (
        6 * forward_flops * alignment_samples * n_transfer_sources
    )
    
    # Parameter transfer: selective copying based on alignment
    transfer_flops = 4 * n_params * n_transfer_sources
    
    # Fine-tuning after each transfer
    finetune_flops = (
        3 * forward_flops * n_train_samples * finetune_epochs * n_transfer_sources
    )
    
    total_flops = (
        initial_training_flops +
        alignment_flops +
        transfer_flops +
        finetune_flops
    )
    
    return {
        'method': 'GRASP',
        'model_params_millions': n_params / 1e6,
        'inference_gflops': forward_flops / 1e9,
        'initial_training_gflops': initial_training_flops / 1e9,
        'alignment_gflops_total': alignment_flops / 1e9,
        'transfer_gflops_total': transfer_flops / 1e9,
        'finetune_gflops_total': finetune_flops / 1e9,
        'total_gflops': total_flops / 1e9,
        'n_sources': n_sources,
        'breakdown': {
            'initial_training_percent': 100 * initial_training_flops / total_flops,
            'alignment_percent': 100 * alignment_flops / total_flops,
            'transfer_percent': 100 * transfer_flops / total_flops,
            'finetune_percent': 100 * finetune_flops / total_flops,
        }
    }


def get_grasp_flops_report(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_train_samples: int,
    initial_epochs: int,
    finetune_epochs: int,
    alignment_batches: int = 1,
    batch_size: int = 32,
    device: torch.device = None,
    **kwargs
) -> Dict:
    """Get GRASP FLOPs report."""
    return count_grasp_total_flops(
        model=model,
        input_size=input_size,
        n_sources=n_sources,
        n_train_samples=n_train_samples,
        initial_epochs=initial_epochs,
        finetune_epochs=finetune_epochs,
        alignment_batches=alignment_batches,
        batch_size=batch_size,
        device=device,
        **kwargs
    )


# ============================================================================
# PEARL FLOPs Calculation
# ============================================================================

def count_pearl_total_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_train_samples: int,
    adapter_epochs: int,
    composition_epochs: int = 3,
    lora_rank: int = 8,
    n_adapted_layers: int = 12,
    batch_size: int = 32,
    device: torch.device = None,
    **kwargs
) -> Dict:
    """
    Calculate total FLOPs for PEARL method.
    
    PEARL: Parameter-Efficient Adapter-based Representation Learning
    
    Phases:
    1. Train LoRA adapters on each source
    2. Learn composition weights for target task
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_params = sum(p.numel() for p in model.parameters())
    forward_flops = count_model_flops(model, input_size, device)
    
    # Estimate total adapter parameters across all adapted layers
    # Each layer gets: rank * (in_features + out_features) for Linear
    # or rank * (in_channels + out_channels) for Conv2d
    avg_layer_params = n_params / 100  # Rough estimate per layer
    adapter_params_per_layer = lora_rank * 2 * avg_layer_params * 0.1
    total_adapter_params = adapter_params_per_layer * n_adapted_layers
    
    # Adapter training for each source (N sources)
    # Training = forward + backward (2x forward) + optimizer step
    adapter_training_single = 3 * forward_flops * n_train_samples * adapter_epochs
    adapter_training_total = n_sources * adapter_training_single
    
    # Composition weight learning
    # Forward passes with all adapters active + learning weights
    # Composition weights are small (N_sources scalars), minimal overhead
    composition_training_flops = (
        3 * forward_flops * n_train_samples * composition_epochs
    )
    
    # Total computation
    total_flops = adapter_training_total + composition_training_flops
    
    return {
        'method': 'PEARL',
        'model_params_millions': n_params / 1e6,
        'adapter_params_millions': total_adapter_params / 1e6,
        'adapter_params_per_source': total_adapter_params,
        'inference_gflops': forward_flops / 1e9,
        'adapter_training_gflops': adapter_training_single / 1e9,
        'adapter_training_gflops_total': adapter_training_total / 1e9,
        'composition_learning_gflops': composition_training_flops / 1e9,
        'total_gflops': total_flops / 1e9,
        'n_sources': n_sources,
        'lora_rank': lora_rank,
        'n_adapted_layers': n_adapted_layers,
        'breakdown': {
            'adapter_training_percent': 100 * adapter_training_total / total_flops,
            'composition_learning_percent': 100 * composition_training_flops / total_flops,
        }
    }


def get_pearl_flops_report(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    n_sources: int,
    n_train_samples: int,
    adapter_epochs: int,
    composition_epochs: int = 3,
    lora_rank: int = 8,
    n_adapted_layers: int = 12,
    batch_size: int = 32,
    device: torch.device = None,
    **kwargs
) -> Dict:
    """Get PEARL FLOPs report."""
    return count_pearl_total_flops(
        model=model,
        input_size=input_size,
        n_sources=n_sources,
        n_train_samples=n_train_samples,
        adapter_epochs=adapter_epochs,
        composition_epochs=composition_epochs,
        lora_rank=lora_rank,
        n_adapted_layers=n_adapted_layers,
        batch_size=batch_size,
        device=device,
        **kwargs
    )

# ============================================================================
# Generic FLOPs Report (for backward compatibility)
# ============================================================================

def get_flops_report(
    model: nn.Module,
    sample_input: torch.Tensor,
    n_train_samples: int,
    n_test_samples: int,
    n_epochs: int,
    method_flops: Dict[str, int] = None
) -> Dict:
    """
    Generate comprehensive FLOPs report for generic methods.
    """
    n_params = sum(p.numel() for p in model.parameters())
    
    train_flops = count_training_flops(model, sample_input, n_train_samples, n_epochs)
    test_flops = count_inference_flops(model, sample_input, n_test_samples)
    
    report = {
        'model_params': n_params,
        'model_params_millions': n_params / 1e6,
        'training_gflops': train_flops / 1e9,
        'inference_gflops': test_flops / 1e9,
    }
    
    if method_flops:
        report['method_flops'] = method_flops
        method_total = sum(method_flops.values())
        report['method_total_gflops'] = method_total / 1e9
    else:
        report['method_total_gflops'] = 0
    
    report['total_gflops'] = (
        train_flops / 1e9 +
        test_flops / 1e9 +
        report['method_total_gflops']
    )
    
    return report


# ============================================================================
# Unified Interface
# ============================================================================

def get_method_flops_report(method: str, **kwargs) -> Dict:
    """
    Get FLOPs report for any supported method.
    """
    method = method.upper().replace('-', '').replace('_', '')
    
    if method == 'ENSEMBLE':
        return get_ensemble_flops_report(**kwargs)
    elif method == 'FUSION':
        return get_fusion_flops_report(**kwargs)
    elif method == 'GRASP':
        return get_grasp_flops_report(**kwargs)
    elif method == 'PEARL':
        return get_pearl_flops_report(**kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported: ENSEMBLE, FUSION, GRASP, PEARL"
        )