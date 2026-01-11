"""
model_clear10.py - PyTorch Lightning Module for CLEAR-10 Multi-Class Classification

This module handles the model architecture, training, and evaluation for CLEAR-10 dataset.
Supports both transformer models (HuggingFace) and CNN models (torchvision).

Each year has 10 classes: baseball, bus, camera, cosplay, dress, hockey, 
laptop, racing, soccer, sweater
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


# ============================================================================
# Model Name Resolution for HuggingFace
# ============================================================================

MODEL_NAME_TO_HF = {
    'mobilevit-xxs': 'apple/mobilevit-xx-small',
    'mobilevit-xs': 'apple/mobilevit-x-small',
    'mobilevit-small': 'apple/mobilevit-small',
    'efficientnet-b1': 'google/efficientnet-b1',
    'resnet-50': 'microsoft/resnet-50',
    'apple/mobilevit-xx-small': 'apple/mobilevit-xx-small',
    'apple/mobilevit-x-small': 'apple/mobilevit-x-small',
    'apple/mobilevit-small': 'apple/mobilevit-small',
    'google/efficientnet-b1': 'google/efficientnet-b1',
    'microsoft/resnet-50': 'microsoft/resnet-50',
}


def resolve_model_name_for_hf(model_name: str) -> str:
    """Resolve model name to valid HuggingFace identifier."""
    if model_name in MODEL_NAME_TO_HF:
        return MODEL_NAME_TO_HF[model_name]
    if '/' in model_name:
        return model_name
    return model_name


# ============================================================================
# Model Architecture Registry
# ============================================================================

MODEL_REGISTRY = {
    'mobilevit-xxs': {
        'type': 'huggingface',
        'model_id': 'apple/mobilevit-xx-small',
        'short_name': 'mvit-xxs',
        'display_name': 'MobileViT-XX-Small',
        'params': 1.3,
        'recommended_lr': 3e-4,
        'recommended_batch_size': 32,
        'description': 'Ultra-lightweight transformer'
    },
    'mobilevit-small': {
        'type': 'huggingface',
        'model_id': 'apple/mobilevit-small',
        'short_name': 'mvit-sm',
        'display_name': 'MobileViT-Small',
        'params': 5.6,
        'recommended_lr': 2e-4,
        'recommended_batch_size': 24,
        'description': 'Larger transformer'
    },
    'resnet-50': {
        'type': 'torchvision',
        'model_id': 'resnet50',
        'short_name': 'resnet50',
        'display_name': 'ResNet-50',
        'params': 25.6,
        'recommended_lr': 1e-4,
        'recommended_batch_size': 32,
        'description': 'Classic CNN architecture',
        'fc_layer_name': 'fc',
        'feature_dim_attr': 'fc'
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
        f"Model '{model_key}' not found in registry.\n"
        f"Available models: {available}\n"
        f"To use a custom model, add it to MODEL_REGISTRY in model_clear10.py\n"
        f"Assuming HuggingFace model. Consider adding it to MODEL_REGISTRY for better integration."
    )
    return None


def load_torchvision_model(model_id: str, num_classes: int, dropout_rate: float = 0.2) -> nn.Module:
    """Load a torchvision model with pretrained weights and replace classification head."""
    if model_id == 'resnet50':
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    else:
        raise ValueError(f"Torchvision model '{model_id}' not supported. "
                        f"Add support in load_torchvision_model() function.")
    
    return model


class CLEAR10Classifier(pl.LightningModule):
    """
    PyTorch Lightning Module for 10-class classification on CLEAR-10 dataset.
    
    Supports both HuggingFace transformers and torchvision CNN architectures.
    """
    
    def __init__(
        self,
        model_name: str = "mobilevit-xxs",
        num_classes: int = 10,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 2,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Get model information from registry
        self.model_info = get_model_info(model_name)
        
        if self.model_info is not None:
            self.model_type = self.model_info['type']
            actual_model_id = self.model_info['model_id']
            self.model_short_name = self.model_info['short_name']
            self.model_display_name = self.model_info['display_name']
        else:
            # Model not in registry, assume HuggingFace
            self.model_type = 'huggingface'
            actual_model_id = resolve_model_name_for_hf(model_name)
            self.model_short_name = model_name.split('/')[-1]
            self.model_display_name = model_name
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Store class names
        if class_names is None:
            self.class_names = [
                'baseball', 'bus', 'camera', 'cosplay',
                'dress', 'hockey', 'laptop', 'racing', 'soccer', 'sweater'
            ]
        else:
            self.class_names = class_names
        
        # Load model based on type
        if self.model_type == 'huggingface':
            self.model = self._load_huggingface_model(
                actual_model_id, num_classes, dropout_rate
            )
        elif self.model_type == 'torchvision':
            self.model = load_torchvision_model(
                actual_model_id, num_classes, dropout_rate
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model loaded: {self.model_display_name}")
        if self.model_info:
            print(f"  Type: {self.model_type}")
            print(f"  Short name: {self.model_short_name} ({self.model_info['params']}M params)")
        print(f"  Model ID: {actual_model_id}")
        print(f"Classification head initialized with {num_classes} classes")
        
        # Set model to training mode
        self.model.train()
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
        
        # Define metrics for multiclass classification
        metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
        })
        
        # Create separate metric collections for train/val/test
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
        # Confusion matrices
        self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # Track best validation metrics
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
    
    def _load_huggingface_model(
        self,
        model_id: str,
        num_classes: int,
        dropout_rate: float
    ) -> nn.Module:
        """Load a HuggingFace model."""
        resolved_id = resolve_model_name_for_hf(model_id)
        
        config = AutoConfig.from_pretrained(resolved_id)
        config.num_labels = num_classes
        
        if hasattr(config, 'image_size'):
            config.image_size = 224
        
        if hasattr(config, 'classifier_dropout') and dropout_rate is not None:
            config.classifier_dropout = dropout_rate
        elif hasattr(config, 'hidden_dropout_prob') and dropout_rate is not None:
            config.hidden_dropout_prob = dropout_rate
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")
            warnings.filterwarnings("ignore", message=".*should probably TRAIN this model.*")
            
            model = AutoModelForImageClassification.from_pretrained(
                resolved_id,
                config=config,
                ignore_mismatched_sizes=True,
            )
        
        return model
    
    def _freeze_backbone(self):
        """Freeze all parameters except the classification head."""
        print("  Freezing backbone (only training classification head)")
        
        if self.model_type == 'huggingface':
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
        
        elif self.model_type == 'torchvision':
            for name, param in self.model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.model_type == 'huggingface':
            outputs = self.model(images)
            return outputs.logits
        else:
            return self.model(images)
    
    def _shared_step(self, batch: tuple, stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        images, labels = batch
        
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        if stage == 'train':
            self.train_metrics.update(preds, labels)
            self.train_confusion_matrix.update(preds, labels)
        elif stage == 'val':
            self.val_metrics.update(preds, labels)
            self.val_confusion_matrix.update(preds, labels)
        elif stage == 'test':
            self.test_metrics.update(preds, labels)
            self.test_confusion_matrix.update(preds, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'preds': preds,
            'labels': labels,
        }
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch."""
        outputs = self._shared_step(batch, 'train')
        loss = outputs['loss']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=len(batch[0]))
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=False, logger=True)
        self.train_metrics.reset()
        self.train_confusion_matrix.reset()
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch."""
        outputs = self._shared_step(batch, 'val')
        loss = outputs['loss']
        
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=len(batch[0]))
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, logger=True)
        
        val_acc = metrics['val_acc']
        val_f1 = metrics['val_f1']
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.log('best_val_acc', self.best_val_acc, prog_bar=False, logger=True)
        
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.log('best_val_f1', self.best_val_f1, prog_bar=False, logger=True)
        
        conf_matrix = self.val_confusion_matrix.compute()
        
        if self.trainer.is_global_zero:
            epoch = self.current_epoch
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Validation Summary:")
            print(f"{'='*80}")
            print(f"  Loss:      {self.trainer.logged_metrics.get('val_loss', 0):.4f}")
            print(f"  Accuracy:  {metrics['val_acc']:.4f} ({metrics['val_acc']*100:.2f}%)")
            print(f"  F1-Score:  {metrics['val_f1']:.4f}")
            print(f"  Precision: {metrics['val_precision']:.4f}")
            print(f"  Recall:    {metrics['val_recall']:.4f}")
            print(f"{'='*80}\n")
        
        self.val_metrics.reset()
        self.val_confusion_matrix.reset()
    
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step for a single batch."""
        outputs = self._shared_step(batch, 'test')
        loss = outputs['loss']
        
        self.log('test_loss', loss, on_step=False, on_epoch=True,
                 batch_size=len(batch[0]))
        
        return loss
    
    def on_test_epoch_end(self):
        """Called at the end of testing."""
        metrics = self.test_metrics.compute()
        conf_matrix = self.test_confusion_matrix.compute()
        
        self.log_dict(metrics, logger=True)
        
        if self.trainer.is_global_zero:
            print("\n" + "=" * 80)
            print("TEST RESULTS")
            print("=" * 80)
            for key, value in metrics.items():
                print(f"  {key:20s}: {value:.4f}")
            print("=" * 80)
        
        self.test_metrics.reset()
        self.test_confusion_matrix.reset()
    
    def predict_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for a single batch."""
        if isinstance(batch, tuple):
            images, _ = batch
        else:
            images = batch
        
        logits = self(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'logits': logits,
        }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            is_classifier = False
            if self.model_type == 'huggingface':
                is_classifier = 'classifier' in name or 'head' in name
            elif self.model_type == 'torchvision':
                is_classifier = 'fc' in name or 'classifier' in name
            
            if is_classifier:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        if len(classifier_params) > 0 and len(backbone_params) > 0:
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': self.hparams.learning_rate},
                {'params': classifier_params, 'lr': self.hparams.learning_rate * 10},
            ], weight_decay=self.hparams.weight_decay)
        elif len(classifier_params) > 0:
            optimizer = torch.optim.AdamW(
                classifier_params,
                lr=self.hparams.learning_rate * 10,
                weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        
        if self.trainer.max_epochs is not None:
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * (self.hparams.warmup_epochs / self.trainer.max_epochs))
        else:
            warmup_steps = 0
            total_steps = 1000
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint."""
        checkpoint['best_val_acc'] = self.best_val_acc
        checkpoint['best_val_f1'] = self.best_val_f1
        checkpoint['class_names'] = self.class_names
        checkpoint['model_short_name'] = self.model_short_name
        checkpoint['model_display_name'] = self.model_display_name
        checkpoint['model_type'] = self.model_type
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        if 'model_type' in checkpoint:
            self.model_type = checkpoint['model_type']