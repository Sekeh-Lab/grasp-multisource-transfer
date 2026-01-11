"""
Metrics calculation and visualization utilities for all experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix as sklearn_confusion_matrix
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None,
    num_classes: int = None
) -> Dict:
    """
    Calculate comprehensive metrics from predictions.
    
    Args:
        y_true: True labels (can be predictions or labels array)
        y_pred: Predicted class labels (can be probabilities array) 
        y_probs: Class probabilities (optional)
        num_classes: Number of classes
    
    Returns:
        Dictionary with all metrics
    """
    # Handle different input formats for backward compatibility
    if y_probs is None and len(y_pred.shape) > 1:
        # y_pred is actually probabilities
        y_probs = y_pred
        y_pred = np.argmax(y_probs, axis=1)
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
    
    # Infer num_classes if not provided
    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # AUROC
    auroc = 0.0
    if y_probs is not None:
        try:
            if num_classes == 2:
                # Binary classification
                if len(y_probs.shape) > 1 and y_probs.shape[1] >= 2:
                    auroc = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    auroc = roc_auc_score(y_true, y_probs)
            else:
                # Multi-class classification
                auroc = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='weighted'
                )
        except Exception as e:
            print(f"Warning: Could not calculate AUROC: {e}")
            auroc = 0.0
    
    # Confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auroc': float(auroc),
        'confusion_matrix': cm.tolist()
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (10, 8)
):
    """
    Save confusion matrix visualization.
    """
    # Calculate confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    else:
        cm_normalized = cm
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to: {save_path}")


def save_metrics(
    metrics: Dict,
    save_path: Path,
    pretty: bool = True
):
    """
    Save metrics to JSON file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if pretty:
            json.dump(metrics, f, indent=2)
        else:
            json.dump(metrics, f)


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def print_metrics_summary(metrics: Dict, prefix: str = ""):
    """Print formatted metrics summary."""
    print(f"\n{prefix}Metrics Summary:")
    print(f"{prefix}{'='*50}")
    print(f"{prefix}  Accuracy:  {metrics['accuracy']*100:6.2f}%")
    print(f"{prefix}  Precision: {metrics['precision']:6.4f}")
    print(f"{prefix}  Recall:    {metrics['recall']:6.4f}")
    print(f"{prefix}  F1-Score:  {metrics['f1_score']:6.4f}")
    
    if 'auroc' in metrics and metrics['auroc'] > 0:
        print(f"{prefix}  AUROC:     {metrics['auroc']:6.4f}")
    print(f"{prefix}{'='*50}")


def compare_methods(
    results_dict: Dict[str, Dict],
    save_path: Optional[Path] = None
):
    """
    Create comparison plot for multiple methods.
    """
    methods = list(results_dict.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    data = {
        'Accuracy': [results_dict[m]['accuracy'] for m in methods],
        'Precision': [results_dict[m]['precision'] for m in methods],
        'Recall': [results_dict[m]['recall'] for m in methods],
        'F1-Score': [results_dict[m]['f1_score'] for m in methods],
    }
    
    x = np.arange(len(methods))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics_names):
        ax.bar(x + i * width, data[metric], width, label=metric)
    
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_training_curves(
    history: Dict,
    save_path: Path,
    title: str = "Training Curves"
):
    """
    Create training/validation curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy curve
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()