"""
Shared data loading utilities for all experiments.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

# Add baseline training to path for dataset modules
baseline_path = Path(__file__).parent.parent.parent / 'baseline_training'
baseline_path_str = str(baseline_path.resolve())
if baseline_path_str not in sys.path:
    sys.path.insert(0, baseline_path_str)


def get_data_module(
    dataset: str,
    target: str,
    data_root: str = "../../datasets",
    batch_size: int = 32,
    num_workers: int = 0,
    model_name: str = "apple/mobilevit-xx-small"
) -> Dict:
    """
    Get data module for specified dataset.
    """    
    if dataset == 'yearbook':
        try:
            from yearbook.data_module_yearbook import YearbookDataModule
        except ImportError as e:
            raise ImportError(
                f"Failed to import YearbookDataModule. "
                f"Ensure baseline_training/yearbook exists and is accessible. "
                f"Current sys.path includes: {baseline_path_str}"
            ) from e
        
        data_module = YearbookDataModule(
            data_root=str(Path(data_root) / 'Yearbook_Decades'),
            subset_name=target,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            augmentation=True
        )
        data_module.prepare_data()
        data_module.setup()
        
        return {
            'train': data_module.train_dataloader(),
            'val': data_module.val_dataloader(),
            'test': data_module.test_dataloader(),
            'num_classes': 2,
            'class_names': data_module.get_dataset_stats()['class_names']
        }
    
    elif dataset == 'clear10':
        try:
            from clear10.data_module_clear10 import CLEAR10DataModule
        except ImportError as e:
            raise ImportError(
                f"Failed to import CLEAR10DataModule. "
                f"Ensure baseline_training/clear10 exists and is accessible. "
                f"Current sys.path includes: {baseline_path_str}"
            ) from e
        
        data_module = CLEAR10DataModule(
            data_root=str(Path(data_root) / 'CLEAR10'),
            year_name=target,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            augmentation=True
        )
        data_module.prepare_data()
        data_module.setup()
        
        return {
            'train': data_module.train_dataloader(),
            'val': data_module.val_dataloader(),
            'test': data_module.test_dataloader(),
            'num_classes': 10,
            'class_names': data_module.get_dataset_stats()['class_names']
        }
    
    elif dataset == 'clear100':
        try:
            from clear100_30classes.data_module_clear100 import CLEAR100DataModule
        except ImportError as e:
            raise ImportError(
                f"Failed to import CLEAR100DataModule. "
                f"Ensure baseline_training/clear100_30classes exists and is accessible. "
                f"Current sys.path includes: {baseline_path_str}"
            ) from e
        
        data_module = CLEAR100DataModule(
            data_root=str(Path(data_root) / 'CLEAR100_30classes'),
            year_name=target,
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            augmentation=True
        )
        data_module.prepare_data()
        data_module.setup()
        
        return {
            'train': data_module.train_dataloader(),
            'val': data_module.val_dataloader(),
            'test': data_module.test_dataloader(),
            'num_classes': 30,
            'class_names': data_module.get_dataset_stats()['class_names']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: yearbook, clear10, clear100, genimage")


def get_dataloaders(
    dataset: str,
    target: str,
    data_root: str = "../../datasets",
    batch_size: int = 32,
    num_workers: int = 0,
    model_name: str = "apple/mobilevit-xx-small"
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Get dataloaders for specified dataset.
    
    Convenience wrapper around get_data_module() that returns dataloaders
    in tuple format for backward compatibility.
    """
    data_dict = get_data_module(
        dataset=dataset,
        target=target,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name
    )
    
    return (
        data_dict['train'],
        data_dict['val'],
        data_dict['test'],
        data_dict['num_classes']
    )