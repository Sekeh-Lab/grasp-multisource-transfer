"""
Shared utilities for sequential transfer learning experiments.

ICPR 2026 Submission - Anonymous Version
RRPR Badge: Reproducibility Compliant

This module provides shared functionality for GRASP, PEARL, FUSION, ENSEMBLE
and other sequential transfer learning methods across multiple datasets.
"""

__version__ = '1.0.0'
__author__ = 'Anonymous Authors'
__license__ = 'MIT'

# Data utilities
from .data_utils import get_data_module, get_dataloaders

# Model utilities
from .model_utils import (
    load_model_from_checkpoint,
    load_baseline_model,
    get_checkpoint_path,
    get_model_info,
    standardize_model_name  # NEW: For consistent model naming across all experiments
)

# FLOPs utilities
from .flops_counter import (
    count_flops,
    count_merging_flops,
    count_training_flops,
    count_inference_flops,
    get_flops_report,
    get_grasp_flops_report,
    get_pearl_flops_report
)

# Metrics utilities
from .metrics_utils import (
    calculate_metrics,
    save_confusion_matrix,
    save_metrics,
    load_metrics,
    print_metrics_summary,
    compare_methods,
    create_training_curves
)

__all__ = [
    # Data utilities
    'get_data_module',
    'get_dataloaders',
    
    # Model utilities
    'load_model_from_checkpoint',
    'load_baseline_model',
    'get_checkpoint_path',
    'get_model_info',
    'standardize_model_name',  # NEW: For consistent model naming
    
    # FLOPs utilities
    'count_flops',
    'count_merging_flops',
    'count_training_flops',
    'count_inference_flops',
    'get_flops_report',
    'get_grasp_flops_report',
    'get_pearl_flops_report',
    
    # Metrics utilities
    'calculate_metrics',
    'save_confusion_matrix',
    'save_metrics',
    'load_metrics',
    'print_metrics_summary',
    'compare_methods',
    'create_training_curves',
]
