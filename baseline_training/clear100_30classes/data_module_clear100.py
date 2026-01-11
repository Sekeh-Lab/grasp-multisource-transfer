"""
data_module_clear100.py - PyTorch Lightning DataModule for CLEAR-100 Dataset

This module handles all data loading, preprocessing, and augmentation for the CLEAR-100 
dataset (30-class subset). Supports both single years and merged years.

30 classes: airplane, aquarium, baseball, beer, billiard, boat, bowling_ball, bridge,
           bus, camera, castle, chocolate, coins, diving, field_hockey, food_truck,
           football, guitar, hair_salon, helicopter, horse_riding, motorcycle,
           pet_store, racing_car, shopping_mall, skyscraper, soccer, stadium, train,
           video_game
"""

import os
from pathlib import Path
from typing import Optional, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor


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
    print(f"Warning: Unknown model name '{model_name}', using as-is")
    return model_name


# ============================================================================
# Torchvision Model Registry
# ============================================================================

TORCHVISION_MODELS = {
    'resnet50', 'resnet-50', 'efficientnet'
}


def is_torchvision_model(model_name: str) -> bool:
    """Check if a model should use torchvision preprocessing."""
    model_lower = model_name.lower().replace('_', '-')
    
    if model_lower in TORCHVISION_MODELS:
        return True
    
    for tv_model in TORCHVISION_MODELS:
        if tv_model in model_lower:
            return True
    
    return False


try:
    from model_clear100 import MODEL_REGISTRY
except ImportError:
    MODEL_REGISTRY = {
        'mobilevit-xxs': {'model_id': 'apple/mobilevit-xx-small'},
        'mobilevit-small': {'model_id': 'apple/mobilevit-small'},
    }


def resolve_model_name(model_name: str) -> str:
    """Resolve a model name to its HuggingFace model ID."""
    if model_name in MODEL_NAME_TO_HF:
        return MODEL_NAME_TO_HF[model_name]
    
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]['model_id']
    
    if '/' in model_name:
        return model_name
    
    for info in MODEL_REGISTRY.values():
        if info['model_id'] == model_name:
            return model_name
    
    return model_name


class CLEAR100DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CLEAR-100 dataset (30 classes).
    
    Supports both HuggingFace transformers and torchvision CNNs with
    automatic preprocessing detection.
    
    Supports both single years and merged years:
    - Single: year_1, year_2, ..., year_11
    - Merged: year_1-2, year_3-4, year_5-6, year_7-8, year_9-10
    
    Args:
        data_root: Root directory containing the CLEAR100 dataset
        year_name: Name of the year to use
        model_name: Model name/identifier for preprocessing
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        augmentation: Whether to use data augmentation during training
    """
    
    NUM_CLASSES = 30
    CLASS_NAMES = [
        'airplane', 'aquarium', 'baseball', 'beer', 'billiard',
        'boat', 'bowling_ball', 'bridge', 'bus', 'camera',
        'castle', 'chocolate', 'coins', 'diving', 'field_hockey',
        'food_truck', 'football', 'guitar', 'hair_salon', 'helicopter',
        'horse_riding', 'motorcycle', 'pet_store', 'racing_car', 'shopping_mall',
        'skyscraper', 'soccer', 'stadium', 'train', 'video_game'
    ]
    
    def __init__(
        self,
        data_root: str = "../../datasets/CLEAR100_30classes",
        year_name: str = "year_1-2",
        model_name: str = "mobilevit-xxs",
        batch_size: int = 32,
        num_workers: int = -1,
        augmentation: bool = True,
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.year_name = year_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        
        self.year_dir = self.data_root / year_name
        self.train_dir = self.year_dir / "train"
        self.val_dir = self.year_dir / "val"
        self.test_dir = self.year_dir / "test"
        
        self.is_torchvision = is_torchvision_model(model_name)
        
        if self.is_torchvision:
            print(f"  Detected torchvision model: Using standard ImageNet preprocessing")
            self._setup_torchvision_preprocessing()
        else:
            print(f"  Detected HuggingFace model: Using AutoImageProcessor")
            self._setup_huggingface_preprocessing()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _setup_torchvision_preprocessing(self):
        """Setup preprocessing for torchvision models."""
        self.crop_size = 224
        self.resize_size = 256
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.processor = None
        
        print(f"  Image size: {self.crop_size}x{self.crop_size}")
        print(f"  Normalization: ImageNet (mean={self.image_mean[:2]}..., std={self.image_std[:2]}...)")
    
    def _setup_huggingface_preprocessing(self):
        """Setup preprocessing for HuggingFace models."""
        resolved_model = resolve_model_name_for_hf(self.model_name)
        print(f"Resolved model name: '{self.model_name}' -> '{resolved_model}'")
        
        self.processor = AutoImageProcessor.from_pretrained(resolved_model, use_fast=True)
        
        self.crop_size = 224
        self.resize_size = 256
        
        self.image_mean = None
        self.image_std = None
        
        if hasattr(self.processor, 'image_mean') and self.processor.image_mean is not None:
            self.image_mean = self.processor.image_mean
        if hasattr(self.processor, 'image_std') and self.processor.image_std is not None:
            self.image_std = self.processor.image_std
        
        if self.image_mean is None and hasattr(self.processor, 'image_processor'):
            self.image_mean = getattr(self.processor.image_processor, 'image_mean', None)
            self.image_std = getattr(self.processor.image_processor, 'image_std', None)
        
        if self.image_mean is None:
            self.image_mean = [0.485, 0.456, 0.406]
            self.image_std = [0.229, 0.224, 0.225]
            print(f"  Using default ImageNet normalization")
        
        print(f"  Image size: {self.crop_size}x{self.crop_size}")
        print(f"  Normalization: mean={self.image_mean[:2]}..., std={self.image_std[:2]}...")
    
    def get_transforms(self, train: bool = False) -> transforms.Compose:
        """Get image transforms for training or validation/test."""
        if train and self.augmentation:
            transform_list = [
                transforms.Resize(self.resize_size),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        else:
            transform_list = [
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        
        return transforms.Compose(transform_list)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageFolder(
                root=str(self.train_dir),
                transform=self.get_transforms(train=True)
            )
            
            self.val_dataset = ImageFolder(
                root=str(self.val_dir),
                transform=self.get_transforms(train=False)
            )
            
            print(f"\nDataset loaded:")
            print(f"  Train: {len(self.train_dataset)} images")
            print(f"  Val:   {len(self.val_dataset)} images")
            print(f"  Classes: {self.NUM_CLASSES}")
        
        if stage == 'test' or stage is None:
            self.test_dataset = ImageFolder(
                root=str(self.test_dir),
                transform=self.get_transforms(train=False)
            )
            
            print(f"  Test:  {len(self.test_dataset)} images")
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def get_class_names(self) -> List[str]:
        """Get the class names."""
        return self.CLASS_NAMES
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return self.NUM_CLASSES
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics including size and class distribution."""
        stats = {
            'year_name': self.year_name,
            'num_classes': self.NUM_CLASSES,
            'class_names': self.CLASS_NAMES,
            'crop_size': self.crop_size,
        }
        
        if self.train_dataset:
            stats['train_size'] = len(self.train_dataset)
            train_class_counts = {}
            for _, label in self.train_dataset.samples:
                class_name = self.CLASS_NAMES[label]
                train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1
            stats['train_class_distribution'] = train_class_counts
        
        if self.val_dataset:
            stats['val_size'] = len(self.val_dataset)
            val_class_counts = {}
            for _, label in self.val_dataset.samples:
                class_name = self.CLASS_NAMES[label]
                val_class_counts[class_name] = val_class_counts.get(class_name, 0) + 1
            stats['val_class_distribution'] = val_class_counts
        
        if self.test_dataset:
            stats['test_size'] = len(self.test_dataset)
            test_class_counts = {}
            for _, label in self.test_dataset.samples:
                class_name = self.CLASS_NAMES[label]
                test_class_counts[class_name] = test_class_counts.get(class_name, 0) + 1
            stats['test_class_distribution'] = test_class_counts
        
        return stats


def auto_detect_num_workers(verbose: bool = True) -> int:
    """Automatically detect number of data loading workers."""
    import os
    import psutil
    
    if verbose:
        print("Auto-detecting number of data loading workers...")
    
    cpu_count = os.cpu_count() or 1
    memory = psutil.virtual_memory()
    total_ram_gb = memory.total / (1024**3)
    available_ram_gb = memory.available / (1024**3)
    
    shm_stats = psutil.disk_usage('/dev/shm')
    shm_size_gb = shm_stats.total / (1024**3)
    
    if verbose:
        print(f"System resources:")
        print(f"   - CPUs: {cpu_count}")
        print(f"   - Total RAM: {total_ram_gb:.1f} GB")
        print(f"   - Available RAM: {available_ram_gb:.1f} GB")
        print(f"   - Shared memory: {shm_size_gb:.2f} GB")
    
    if shm_size_gb < 1.0:
        if verbose:
            print(f"   Using num_workers=0 (insufficient shared memory)")
        return 0
    elif available_ram_gb < 4.0:
        if verbose:
            print(f"   Using num_workers=2 (limited RAM)")
        return 2
    else:
        optimal = min(cpu_count - 2, 8)
        optimal = max(optimal, 2)
        if verbose:
            print(f"   Using num_workers={optimal}")
        return optimal