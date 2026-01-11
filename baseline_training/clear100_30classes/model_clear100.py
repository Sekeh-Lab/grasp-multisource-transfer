"""
CLEAR-100 classifier: PyTorch Lightning module for 30-class classification.
Supports 4 architectures: mobilevit-xxs, mobilevit-xs, resnet-50, efficientnet-b1.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoConfig
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
from typing import Dict, Any, Optional, List
import warnings
import torchvision.models as tv_models


# Model name resolution for HuggingFace - ONLY 4 models supported
MODEL_NAME_TO_HF = {
    'mobilevit-xxs': 'apple/mobilevit-xx-small',
    'mobilevit-xs': 'apple/mobilevit-x-small',
    'resnet-50': 'microsoft/resnet-50',
    'efficientnet-b1': 'google/efficientnet-b1',
    'apple/mobilevit-xx-small': 'apple/mobilevit-xx-small',
    'apple/mobilevit-x-small': 'apple/mobilevit-x-small',
    'microsoft/resnet-50': 'microsoft/resnet-50',
    'google/efficientnet-b1': 'google/efficientnet-b1',
}


def resolve_model_name_for_hf(model_name: str) -> str:
    """Resolve model name to valid HuggingFace identifier."""
    if model_name in MODEL_NAME_TO_HF:
        return MODEL_NAME_TO_HF[model_name]
    if '/' in model_name:
        return model_name
    return model_name


# Model architecture registry - ONLY 4 models for paper
MODEL_REGISTRY = {
    'mobilevit-xxs': {
        'type': 'huggingface',
        'model_id': 'apple/mobilevit-xx-small',
        'short_name': 'mvit-xxs',
        'display_name': 'MobileViT-XXS',
        'params': 1.3,
        'recommended_lr': 3e-4,
        'recommended_batch_size': 32,
    },
    'mobilevit-xs': {
        'type': 'huggingface',
        'model_id': 'apple/mobilevit-x-small',
        'short_name': 'mvit-xs',
        'display_name': 'MobileViT-XS',
        'params': 2.3,
        'recommended_lr': 3e-4,
        'recommended_batch_size': 32,
    },
    'resnet-50': {
        'type': 'torchvision',
        'model_id': 'resnet50',
        'short_name': 'resnet50',
        'display_name': 'ResNet-50',
        'params': 25.6,
        'recommended_lr': 1e-4,
        'recommended_batch_size': 32,
        'fc_layer_name': 'fc',
        'feature_dim_attr': 'fc'
    },
    'efficientnet-b1': {
        'type': 'huggingface',
        'model_id': 'google/efficientnet-b1',
        'short_name': 'effnet-b1',
        'display_name': 'EfficientNet-B1',
        'params': 7.8,
        'recommended_lr': 2e-4,
        'recommended_batch_size': 32,
    },
}


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model information from registry."""
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    
    for key, info in MODEL_REGISTRY.items():
        if info['model_id'] == model_key:
            return info
    
    available = ', '.join(MODEL_REGISTRY.keys())
    warnings.warn(
        f"Model '{model_key}' not in registry. Available: {available}"
    )
    return None


def load_torchvision_model(model_id: str, num_classes: int, dropout_rate: float = 0.2) -> nn.Module:
    """Load a torchvision model with pretrained weights."""
    if model_id == 'resnet50':
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported torchvision model: {model_id}")
    
    return model


class CLEAR100Classifier(pl.LightningModule):
    """
    CLEAR-100 multi-class classifier (30 classes).
    Supports HuggingFace transformers and torchvision CNNs.
    """
    
    def __init__(
        self,
        model_name: str = "mobilevit-xxs",
        num_classes: int = 30,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 2,
        max_epochs: int = 15,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        
        # Get model info
        self.model_info = get_model_info(model_name)
        
        # Load model
        if self.model_info and self.model_info['type'] == 'torchvision':
            self.model = load_torchvision_model(
                self.model_info['model_id'],
                num_classes,
                dropout_rate
            )
        else:
            resolved_model_name = resolve_model_name_for_hf(model_name)
            config = AutoConfig.from_pretrained(resolved_model_name)
            config.num_labels = num_classes
            self.model = AutoModelForImageClassification.from_pretrained(
                resolved_model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Metrics
        metrics = MetricCollection({
            'acc': Accuracy(task='multiclass', num_classes=num_classes),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
        })
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
        self.val_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _freeze_backbone(self):
        """Freeze backbone layers."""
        if self.model_info and self.model_info['type'] == 'torchvision':
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        else:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
    
    def forward(self, pixel_values):
        """Forward pass."""
        if self.model_info and self.model_info['type'] == 'torchvision':
            return self.model(pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
            return outputs.logits
    
    def _shared_step(self, batch, metrics):
        """Shared step for train/val/test."""
        pixel_values, labels = batch
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, labels)
        
        return loss, preds, labels
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, preds, labels = self._shared_step(batch, self.train_metrics)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds, labels = self._shared_step(batch, self.val_metrics)
        self.val_confusion.update(preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()
        self.val_confusion.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds, labels = self._shared_step(batch, self.test_metrics)
        self.test_confusion.update(preds, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_confusion.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Linear warmup + cosine decay
        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            
            progress = float(current_epoch - self.warmup_epochs) / float(
                max(1, self.max_epochs - self.warmup_epochs)
            )
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'total_mb': size_mb,
            'param_mb': param_size / 1024 / 1024,
            'buffer_mb': buffer_size / 1024 / 1024,
        }
