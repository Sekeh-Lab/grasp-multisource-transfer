"""
Shared model loading utilities.
"""

import sys
import traceback
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForImageClassification

# Add baseline_training to path
sys.path.append(str(Path(__file__).parent.parent.parent / "baseline_training"))

# Model name standardization mapping (used for checkpoints AND result paths)
MODEL_NAME_MAP = {
    "apple/mobilevit-xx-small": "mobilevit-xxs",
    "mobilevit-xx-small": "mobilevit-xxs",
    "mobilevit-xxs": "mobilevit-xxs",
    "apple/mobilevit-x-small": "mobilevit-xs",
    "mobilevit-x-small": "mobilevit-xs",
    "mobilevit-xs": "mobilevit-xs",
    "apple/mobilevit-small": "mobilevit-small",
    "mobilevit-small": "mobilevit-small",
    "google/efficientnet-b1": "efficientnet-b1",
    "efficientnet-b1": "efficientnet-b1",
    "microsoft/resnet-50": "resnet-50",
    "resnet-50": "resnet-50",
    "resnet50": "resnet-50",
}

# Reverse mapping: from standardized name back to HuggingFace model identifier
STANDARD_TO_HF_MAP = {
    "mobilevit-xxs": "apple/mobilevit-xx-small",
    "mobilevit-xs": "apple/mobilevit-x-small",
    "mobilevit-small": "apple/mobilevit-small",
    "efficientnet-b1": "google/efficientnet-b1",
    "resnet-50": "microsoft/resnet-50",
}


def standardize_model_name(model_name: str) -> str:
    """
    Standardize model name to match checkpoint and results naming convention.
    """
    if model_name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model_name]
    standardized = (
        model_name.replace("/", "-")
        .replace("apple-", "")
        .replace("microsoft-", "")
        .replace("google-", "")
    )
    return standardized


def get_huggingface_model_name(model_name: str) -> str:
    """
    Convert any model name format to valid HuggingFace identifier.

    Full HuggingFace names stay as-is.
    Standardized names get converted back to HuggingFace format.
    """
    if "/" in model_name:
        return model_name

    if model_name in STANDARD_TO_HF_MAP:
        return STANDARD_TO_HF_MAP[model_name]

    return model_name


def _is_torchvision_backbone(model_name: str) -> bool:
    """Return True if this model should be loaded from torchvision, not HF."""
    name = standardize_model_name(model_name)
    return name in {"vgg16-bn", "resnet-50"}


def _build_torchvision_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Build a torchvision backbone with a num_classes head.
    """
    name = standardize_model_name(model_name)

    if name == "resnet-50":
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnet50(pretrained=True)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone

    raise ValueError(f"Unknown torchvision model name: {model_name}")


def _get_dataset_checkpoint_dir(dataset: str, base_dir: str) -> Path:
    """Get the correct checkpoint directory for each dataset."""
    base_path = Path(base_dir)
    
    if dataset == "clear100":
        return base_path / "clear100_30classes" / "model_checkpoints"
    else:
        return base_path / dataset / "model_checkpoints"


def load_baseline_model(
    dataset: str,
    subset: str,
    model_name: str,
    num_classes: int,
    checkpoint_dir: str = "../../baseline_training",
) -> Optional[nn.Module]:
    """Load a pre-trained baseline model from checkpoint."""
    model_prefix = standardize_model_name(model_name)
    checkpoint_path = _get_dataset_checkpoint_dir(dataset, checkpoint_dir)
    subset_clean = subset.replace("/", "-")

    pattern = f"{model_prefix}_{subset_clean}-best-acc*.ckpt"
    checkpoint_files = list(checkpoint_path.glob(pattern))

    if not checkpoint_files:
        print(f"ERROR: No checkpoint found for pattern: {pattern}")
        print(f"Searched in: {checkpoint_path}")
        return None

    checkpoint_file = checkpoint_files[0]
    print(f"Loading checkpoint: {checkpoint_file.name}")

    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        if _is_torchvision_backbone(model_name):
            model = _build_torchvision_model(model_name, num_classes)
        else:
            hf_model_name = get_huggingface_model_name(model_name)
            model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "", 1)
            else:
                new_key = key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print(f"Successfully loaded checkpoint for {subset}")
        return model

    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        traceback.print_exc()
        return None


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """Load model from checkpoint with metadata."""
    if "yearbook" in str(checkpoint_path).lower() or num_classes == 2:
        from yearbook.model_yearbook import YearbookDecadeClassifier
        ModelClass = YearbookDecadeClassifier
    elif "clear10" in str(checkpoint_path).lower() or num_classes == 10:
        from clear10.model_clear10 import CLEAR10Classifier
        ModelClass = CLEAR10Classifier
    elif "clear100" in str(checkpoint_path).lower() or num_classes == 30:
        from clear100_30classes.model_clear100 import CLEAR100Classifier
        ModelClass = CLEAR100Classifier
    else:
        print(f"ERROR loading model from checkpoint. Check the checkpoint path and folder name.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ModelClass(model_name=model_name, num_classes=num_classes)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key.replace("model.", "", 1)
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, checkpoint


def get_checkpoint_path(
    dataset: str,
    subset: str,
    model_name: str,
    checkpoint_dir: str = "../../baseline_training",
) -> Optional[Path]:
    """Find checkpoint path for a given dataset and subset."""
    model_prefix = standardize_model_name(model_name)
    checkpoint_path = _get_dataset_checkpoint_dir(dataset, checkpoint_dir)
    subset_clean = subset.replace("/", "-")

    pattern = f"{model_prefix}_{subset_clean}-best-acc*.ckpt"
    checkpoint_files = list(checkpoint_path.glob(pattern))
    if checkpoint_files:
        return checkpoint_files[0]
    return None


def get_model_info(model: nn.Module) -> Dict:
    """Get model information including parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }