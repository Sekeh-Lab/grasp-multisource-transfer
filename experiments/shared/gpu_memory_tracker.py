"""
GPU Memory Tracking Utility for Sequential Transfer Learning Experiments
"""

import torch
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import json
import subprocess


class GPUMemoryTracker:
    """
    Track GPU memory usage throughout experiments.
    
    Records memory at checkpoints and provides comprehensive statistics.
    Works on both CUDA and CPU (returns zeros for CPU).
    """
    
    def __init__(self, device: torch.device, query_nvidia_smi: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            device: Torch device (cuda or cpu)
            query_nvidia_smi: Whether to query nvidia-smi for actual GPU memory
        """
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.query_nvidia_smi = query_nvidia_smi and self.is_cuda
        self.checkpoints = {}
        self.baseline_overhead_mb = 0
        
        if self.is_cuda:
            self.gpu_id = device.index if device.index is not None else 0
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            self.baseline_overhead_mb = self._measure_baseline_overhead()
        else:
            self.gpu_id = None
    
    def _query_nvidia_smi_memory(self) -> float:
        """Query current GPU memory from nvidia-smi."""
        if not self.query_nvidia_smi:
            return 0.0
        
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
                f'--id={self.gpu_id}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            return 0.0
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            return 0.0
    
    def _measure_baseline_overhead(self) -> float:
        """Measure CUDA context and driver overhead."""
        if not self.is_cuda:
            return 0.0
        
        torch.cuda.synchronize(self.device)
        nvidia_mem = self._query_nvidia_smi_memory()
        pytorch_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        
        overhead = nvidia_mem - pytorch_mem
        return max(overhead, 0)
    
    def checkpoint(self, name: str):
        """
        Record memory at a checkpoint.
        
        Args:
            name: Checkpoint name for identification
        """
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            
            allocated_bytes = torch.cuda.memory_allocated(self.device)
            reserved_bytes = torch.cuda.memory_reserved(self.device)
            max_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
            max_reserved_bytes = torch.cuda.max_memory_reserved(self.device)
            
            allocated_mb = allocated_bytes / (1024 * 1024)
            reserved_mb = reserved_bytes / (1024 * 1024)
            max_allocated_mb = max_allocated_bytes / (1024 * 1024)
            max_reserved_mb = max_reserved_bytes / (1024 * 1024)
            
            nvidia_memory_mb = self._query_nvidia_smi_memory()
            estimated_actual = allocated_mb + self.baseline_overhead_mb
            actual_memory_mb = nvidia_memory_mb if nvidia_memory_mb > 0 else estimated_actual
            
            self.checkpoints[name] = {
                'pytorch_allocated_mb': allocated_mb,
                'pytorch_reserved_mb': reserved_mb,
                'pytorch_max_allocated_mb': max_allocated_mb,
                'pytorch_max_reserved_mb': max_reserved_mb,
                'actual_gpu_memory_mb': actual_memory_mb,
                'baseline_overhead_mb': self.baseline_overhead_mb,
            }
        else:
            self.checkpoints[name] = {
                'pytorch_allocated_mb': 0,
                'pytorch_reserved_mb': 0,
                'pytorch_max_allocated_mb': 0,
                'pytorch_max_reserved_mb': 0,
                'actual_gpu_memory_mb': 0,
                'baseline_overhead_mb': 0,
            }
    
    def get_peak_memory(self) -> Dict:
        """Get peak memory usage across all checkpoints."""
        if not self.is_cuda or not self.checkpoints:
            return {
                'pytorch_peak_allocated_mb': 0,
                'pytorch_peak_reserved_mb': 0,
                'actual_peak_gpu_memory_mb': 0,
                'baseline_overhead_mb': 0,
            }
        
        pytorch_peak_allocated = max(
            cp['pytorch_max_allocated_mb'] for cp in self.checkpoints.values()
        )
        pytorch_peak_reserved = max(
            cp['pytorch_reserved_mb'] for cp in self.checkpoints.values()
        )
        actual_peak = max(
            cp['actual_gpu_memory_mb'] for cp in self.checkpoints.values()
        )
        
        return {
            'pytorch_peak_allocated_mb': pytorch_peak_allocated,
            'pytorch_peak_allocated_gb': pytorch_peak_allocated / 1024,
            'pytorch_peak_reserved_mb': pytorch_peak_reserved,
            'pytorch_peak_reserved_gb': pytorch_peak_reserved / 1024,
            'actual_peak_gpu_memory_mb': actual_peak,
            'actual_peak_gpu_memory_gb': actual_peak / 1024,
            'baseline_overhead_mb': self.baseline_overhead_mb,
            'baseline_overhead_gb': self.baseline_overhead_mb / 1024,
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive memory summary."""
        return {
            'checkpoints': self.checkpoints,
            'peak': self.get_peak_memory(),
            'device': str(self.device),
            'gpu_id': self.gpu_id,
            'is_cuda': self.is_cuda,
            'baseline_overhead_mb': self.baseline_overhead_mb,
        }
    
    def save_profile(self, filepath: Path):
        """Save memory profile to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def print_summary(self, prefix: str = ""):
        """Print formatted memory summary."""
        if self.is_cuda:
            peak = self.get_peak_memory()
            print(f"{prefix}GPU Memory Usage:")
            print(f"{prefix}  Actual GPU Memory: {peak['actual_peak_gpu_memory_mb']:.2f} MB ({peak['actual_peak_gpu_memory_gb']:.4f} GB)")
            print(f"{prefix}  PyTorch Allocated: {peak['pytorch_peak_allocated_mb']:.2f} MB ({peak['pytorch_peak_allocated_gb']:.4f} GB)")
            print(f"{prefix}  PyTorch Reserved:  {peak['pytorch_peak_reserved_mb']:.2f} MB ({peak['pytorch_peak_reserved_gb']:.4f} GB)")
            print(f"{prefix}  Baseline Overhead: {peak['baseline_overhead_mb']:.2f} MB ({peak['baseline_overhead_gb']:.4f} GB)")
        else:
            print(f"{prefix}No GPU used (CPU mode)")
    
    def get_memory_dict(self) -> Dict:
        """Get memory dictionary for results export."""
        peak = self.get_peak_memory()
        return {
            'peak_gpu_memory_mb': peak['actual_peak_gpu_memory_mb'],
            'peak_gpu_memory_gb': peak['actual_peak_gpu_memory_gb'],
            'peak_allocated_mb': peak['pytorch_peak_allocated_mb'],
            'peak_allocated_gb': peak['pytorch_peak_allocated_gb'],
            'peak_reserved_mb': peak['pytorch_peak_reserved_mb'],
            'peak_reserved_gb': peak['pytorch_peak_reserved_gb'],
            'pytorch_peak_allocated_mb': peak['pytorch_peak_allocated_mb'],
            'pytorch_peak_allocated_gb': peak['pytorch_peak_allocated_gb'],
            'pytorch_peak_reserved_mb': peak['pytorch_peak_reserved_mb'],
            'pytorch_peak_reserved_gb': peak['pytorch_peak_reserved_gb'],
            'baseline_overhead_mb': peak['baseline_overhead_mb'],
            'baseline_overhead_gb': peak['baseline_overhead_gb'],
            'device': str(self.device),
            'gpu_id': self.gpu_id,
            'is_cuda': self.is_cuda,
        }


def count_model_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_model_memory_footprint(model: torch.nn.Module, precision: str = 'float32') -> float:
    """
    Calculate model weight memory footprint in MB.
    
    Args:
        model: PyTorch model
        precision: 'float32' or 'float16'
    
    Returns:
        Memory in MB
    """
    num_params = count_model_parameters(model)
    bytes_per_param = 4 if precision == 'float32' else 2
    return (num_params * bytes_per_param) / (1024 * 1024)


def estimate_activation_memory(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device
) -> float:
    """
    Estimate activation memory for a forward pass.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
        device: Device for computation
    
    Returns:
        Estimated activation memory in MB
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    activation_memory = 0
    hooks = []
    
    def hook_fn(module, input, output):
        nonlocal activation_memory
        if isinstance(output, torch.Tensor):
            activation_memory += output.nelement() * output.element_size()
    
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return activation_memory / (1024 * 1024)


def estimate_memory_for_config(
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    num_models: int,
    model: Optional[torch.nn.Module] = None,
    model_params: Optional[int] = None,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'float32',
    device: Optional[torch.device] = None
) -> Dict:
    """
    Estimate GPU memory requirements for a configuration.
    
    Args:
        batch_size: Training/inference batch size
        num_workers: DataLoader num_workers
        prefetch_factor: DataLoader prefetch_factor
        num_models: Number of models loaded
        model: PyTorch model (if available)
        model_params: Model parameter count (if model not provided)
        input_shape: Input image shape (channels, height, width)
        precision: 'float32' or 'float16'
        device: Torch device for activation estimation
    
    Returns:
        Dictionary with detailed memory breakdown
    """
    channels, height, width = input_shape
    bytes_per_element = 4 if precision == 'float32' else 2
    
    elements_per_image = channels * height * width
    bytes_per_image = elements_per_image * bytes_per_element
    bytes_per_batch = bytes_per_image * batch_size
    mb_per_batch = bytes_per_batch / (1024 * 1024)
    
    if num_workers > 0 and prefetch_factor > 0:
        input_batches = 1 + (0.5 * min(prefetch_factor, 2))
    else:
        input_batches = 1
    
    input_memory_mb = mb_per_batch * input_batches
    
    if model is not None:
        model_mb_per_model = get_model_memory_footprint(model, precision)
        if device is not None:
            batch_input_shape = (batch_size, channels, height, width)
            activation_mb_per_model = estimate_activation_memory(model, batch_input_shape, device)
        else:
            activation_mb_per_model = batch_size * 0.6
    else:
        if model_params is None:
            model_params = 2_300_000
        model_bytes = model_params * bytes_per_element
        model_mb_per_model = model_bytes / (1024 * 1024)
        activation_mb_per_model = batch_size * 0.6
    
    total_model_mb = model_mb_per_model * num_models
    activation_memory_mb = activation_mb_per_model * num_models
    
    pytorch_subtotal = input_memory_mb + total_model_mb + activation_memory_mb
    
    cuda_context_mb = 500
    cudnn_workspace_mb = 300
    total_overhead_mb = cuda_context_mb + cudnn_workspace_mb
    
    fragmentation_factor = 0.15
    fragmentation_mb = pytorch_subtotal * fragmentation_factor
    
    total_actual_mb = pytorch_subtotal + total_overhead_mb + fragmentation_mb
    
    return {
        'config': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'num_models': num_models,
            'input_shape': input_shape,
            'precision': precision,
        },
        'breakdown_mb': {
            'input_batches': input_memory_mb,
            'model_weights': total_model_mb,
            'activations': activation_memory_mb,
            'pytorch_subtotal': pytorch_subtotal,
            'cuda_context': cuda_context_mb,
            'cudnn_workspace': cudnn_workspace_mb,
            'fragmentation': fragmentation_mb,
        },
        'total_mb': total_actual_mb,
        'total_gb': total_actual_mb / 1024,
        'pytorch_visible_mb': pytorch_subtotal,
        'pytorch_visible_gb': pytorch_subtotal / 1024,
        'overhead_mb': total_overhead_mb + fragmentation_mb,
        'overhead_gb': (total_overhead_mb + fragmentation_mb) / 1024,
    }


def print_memory_estimate(
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    num_models: int,
    model: Optional[torch.nn.Module] = None,
    model_params: Optional[int] = None,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'float32',
    device: Optional[torch.device] = None
):
    """Print formatted memory estimate."""
    est = estimate_memory_for_config(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        num_models=num_models,
        model=model,
        model_params=model_params,
        input_shape=input_shape,
        precision=precision,
        device=device
    )
    
    print(f"\nMemory Estimate")
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Prefetch factor: {prefetch_factor}")
    print(f"  Models loaded: {num_models}")
    print(f"  Precision: {precision}")
    print(f"Memory Breakdown:")
    print(f"  Input batches:        {est['breakdown_mb']['input_batches']:10.2f} MB")
    print(f"  Model weights:        {est['breakdown_mb']['model_weights']:10.2f} MB")
    print(f"  Activations:          {est['breakdown_mb']['activations']:10.2f} MB")
    print(f"  PyTorch subtotal:     {est['breakdown_mb']['pytorch_subtotal']:10.2f} MB")
    print(f"  CUDA context:         {est['breakdown_mb']['cuda_context']:10.2f} MB")
    print(f"  cuDNN workspace:      {est['breakdown_mb']['cudnn_workspace']:10.2f} MB")
    print(f"  Fragmentation:        {est['breakdown_mb']['fragmentation']:10.2f} MB")
    print(f"  Total GPU Memory:     {est['total_mb']:10.2f} MB ({est['total_gb']:.3f} GB)\n")


def generate_memory_report(
    batch_sizes: List[int],
    num_workers: int,
    prefetch_factor: int,
    num_models: int,
    model: Optional[torch.nn.Module] = None,
    model_params: Optional[int] = None,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'float32',
    device: Optional[torch.device] = None
):
    """Generate comprehensive memory report for multiple batch sizes."""
    print(f"\nGPU Memory Requirements Report")
    print(f"Configuration: num_workers={num_workers}, prefetch_factor={prefetch_factor}")
    print(f"Models loaded: {num_models}\n")
    
    for batch_size in batch_sizes:
        print_memory_estimate(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            num_models=num_models,
            model=model,
            model_params=model_params,
            input_shape=input_shape,
            precision=precision,
            device=device
        )