# GRASP: Gradient-Aligned Sequential Parameter Transfer for Memory-Efficient Multi-Source Learning

Official implementation of "GRASP: Gradient-Aligned Sequential Parameter Transfer for Memory-Efficient Multi-Source Learning"

## Overview

This repository provides a complete experimental framework for multi-source transfer learning on temporal distribution shift benchmarks. We evaluate four transfer learning methods across three datasets with varying levels of temporal drift.

**Methods Implemented:**
- **ENSEMBLE** - Multi-model voting ensemble baseline
- **FUSION** - Parameter averaging baseline (Multi-Source method)
- **GRASP** - Our proposed gradient-aligned sequential parameter transfer
- **PEARL** - Parameter-efficient adapter-based learning baseline

**Datasets:**
- **CLEAR-10**: 10-class object recognition (5 temporal bins)
- **CLEAR-100**: 30-class object recognition (5 temporal bins)
- **Yearbook**: Binary gender classification across decades (4 temporal periods)

**Models:**
- MobileViT-XXS (1.3M parameters)
- MobileViT-XS (2.3M parameters)
- ResNet-50 (25.6M parameters)
- EfficientNet-B1 (7.8M parameters)

## Key Features

- Complete preprocessing pipeline for temporal shift datasets
- Baseline training code with PyTorch Lightning
- Four multi-source transfer learning methods
- Comprehensive evaluation metrics (accuracy, FLOPs, memory, time)
- Shared utilities for data loading, model management, and metrics
- Reproducible experimental setup with detailed documentation

## Repository Structure

```
grasp-multisource-transfer/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── preprocessing/                      # Dataset preprocessing scripts
│   ├── README.md                       # Detailed preprocessing guide
│   ├── clear10_preprocessing.py
│   ├── clear100_preprocessing.py
│   └── yearbook_preprocessing.py
│
├── baseline_training/                  # Baseline model training
│   ├── README.md                       # Detailed training guide
│   ├── clear10/                        # CLEAR-10 training code
│   ├── clear100_30classes/             # CLEAR-100 training code
│   └── yearbook/                       # Yearbook training code
│
└── experiments/                        # Multi-source transfer experiments
    ├── README.md                       # Detailed experiments guide
    ├── shared/                         # Shared utilities
    │   ├── data_utils.py
    │   ├── model_utils.py
    │   ├── flops_counter.py
    │   ├── metrics_utils.py
    │   └── gpu_memory_tracker.py
    ├── ensemble/                       # Ensemble method
    ├── fusion/                         # Fusion method (Multi-Source baseline)
    ├── grasp/                          # GRASP method (ours)
    └── pearl/                          # PEARL method
```

Each subdirectory contains a detailed `README.md` with comprehensive usage instructions.

## Quick Start

### 1. Installation

**Requirements:**
- Python 3.10 or 3.11
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch 2.7.0+ with CUDA support

**Install PyTorch with CUDA support:**

```bash
# For CUDA 12.8+ (RTX 50 series, RTX 40 series)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# For CUDA 11.8 (older GPUs)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

See `requirements.txt` for detailed installation instructions and troubleshooting.

### 2. Dataset Preprocessing

Download and preprocess the datasets:

```bash
cd preprocessing

# Download datasets (see preprocessing/README.md for download links)
# CLEAR-10: https://clear-benchmark.github.io/
# CLEAR-100: https://clear-benchmark.github.io/
# Yearbook: https://shiry.ttic.edu/projects/yearbooks/yearbooks.html

# Run preprocessing scripts
python clear10_preprocessing.py
python clear100_preprocessing.py
python yearbook_preprocessing.py
```

This creates organized train/val/test splits in the `datasets/` directory.

See `preprocessing/README.md` for detailed instructions.

### 3. Train Baseline Models

Train models on individual temporal bins:

```bash
cd baseline_training/clear10

# Train all temporal bins for a single model
python train_all_subsets.py --model_name mobilevit-xxs

# Or train individual bins
python train_clear10.py --year_name year_1-2 --model_name mobilevit-xxs
```

Repeat for CLEAR-100 and Yearbook datasets.

See `baseline_training/README.md` for detailed instructions.

### 4. Run Transfer Learning Experiments

Run multi-source transfer experiments:

```bash
cd experiments/ensemble

# Run all targets with 3 trials each
python run_all_ensemble_clear10.py --num_trials 3 --model mobilevit-xxs

# Or run single experiment
python run_ensemble_experiment.py \
    --dataset clear10 \
    --target year_1-2 \
    --sources year_3-4 year_5-6 year_7-8 year_9-10 \
    --model mobilevit-xxs
```

Repeat for other methods (fusion, grasp, pearl) and datasets.

See `experiments/README.md` for detailed instructions.

## Method Overview

### ENSEMBLE (Baseline)

Combines predictions from multiple source models via soft voting:
- Load pre-trained source models
- Average prediction probabilities
- No additional training required

### FUSION (Multi-Source Baseline)

Averages parameters from source models and fine-tunes:
1. Average parameters from all source models
2. Fine-tune averaged model on target data
3. Single model for inference

**Note:** This method is referred to as "Multi-Source" in the paper.

### GRASP (Our Method)

Sequential transfer with gradient-based parameter selection:
1. Train initial model on first source
2. For each additional source:
   - Compute gradient alignment scores
   - Transfer only aligned parameters
   - Fine-tune on next source
3. Final fine-tuning on target

Key advantage: Selective parameter transfer reduces catastrophic forgetting.

### PEARL (Baseline - Created by us)

Parameter-efficient adapter composition:
1. Train LoRA adapters on each source domain
2. Learn composition weights for adapters
3. Apply composed adapter to target

Key advantage: Low memory footprint during training.

## Dataset Information

### CLEAR-10

- **Classes**: 10 object categories (baseball, bus, camera, cosplay, dress, hockey, laptop, racing, soccer, sweater)
- **Temporal Bins**: 5 two-year periods (year_1-2 through year_9-10)
- **Total Images**: ~30,000 (after preprocessing; ~6,000 per bin)
- **Task**: Multi-class classification with temporal distribution shift

### CLEAR-100

- **Classes**: 30 object categories (subset of CLEAR-100 with strongest temporal patterns)
- **Temporal Bins**: 5 two-year periods (year_1-2 through year_9-10)
- **Total Images**: ~30,000 (after preprocessing; ~6,000 per bin)
- **Task**: Multi-class classification with temporal distribution shift

### Yearbook

- **Classes**: Binary gender classification (male/female)
- **Temporal Periods**: 4 multi-decade periods (before_1950s, 1950s_1960s, 1970s_1980s, 1990s_and_later)
- **Total Images**: ~38,000 portraits
- **Task**: Binary classification with decade-specific visual styles

## Experimental Setup

All experiments use:
- **Train/Val/Test Split**: 80/10/10 (stratified)
- **Input Size**: 224×224 images
- **Normalization**: ImageNet statistics
- **Batch Size**: 32 (default)
- **Metrics**: Accuracy, Precision, Recall, F1-Score, FLOPs, GPU Memory, Time

**Transfer Learning Setup:**
- Leave-one-out: Train on N-1 temporal bins, test on 1 held-out bin
- Multiple trials for statistical significance
- Comprehensive evaluation of computational costs

## File Organization

**Preprocessing:**
- Fixed random seed (42) for reproducible data splits
- 80/10/10 train/val/test stratified splits
- Automatic merging of temporal bins

**Baseline Training:**
- PyTorch Lightning framework
- Automatic checkpointing and logging
- TensorBoard visualization
- Configurable via YAML or command-line

**Experiments:**
- Modular design with shared utilities
- Automatic metrics collection (accuracy, FLOPs, memory, time)
- Comprehensive summary reports
- Results saved in JSON format

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (for MobileViT models)
- RAM: 16GB
- Storage: 50GB (datasets + checkpoints)

**Recommended:**
- GPU: 16GB+ VRAM (for all models)
- RAM: 32GB
- Storage: 100GB

**Tested On:**
- NVIDIA GeForce RTX 5080 Laptop GPU (16GB VRAM)
- CUDA 12.8 / PyTorch 2.7.0
- Ubuntu 22.04 / Windows 11

See `requirements.txt` for detailed hardware specifications.

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python train_clear10.py --batch_size 16

# Use mixed precision
python train_clear10.py --precision 16-mixed
```

**Dataset Not Found:**
```bash
# Verify preprocessing completed
ls datasets/CLEAR10/

# Check paths in scripts
python train_clear10.py --data_root /absolute/path/to/datasets
```

**Baseline Checkpoints Missing:**
```bash
# Train baselines first
cd baseline_training/clear10
python train_all_subsets.py --model_name mobilevit-xxs
```

See individual README files for method-specific troubleshooting.

## Citation
Please also cite the datasets:

**CLEAR:**
```bibtex
@inproceedings{lin2021clear,
  title={CLEAR: A Dataset for Continual LEArning on Real-Robot Sensory Data},
  author={Lin, Zhiqiu and others},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

**Yearbook:**
```bibtex
@inproceedings{ginosar2015century,
  title={A Century of Portraits: A Visual Historical Record of American High School Yearbooks},
  author={Ginosar, Shiry and others},
  booktitle={IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year={2015}
}
```

## License

This code is released for research purposes only. See individual dataset licenses:
- **CLEAR**: https://clear-benchmark.github.io/
- **Yearbook**: https://shiry.ttic.edu/projects/yearbooks/yearbooks.html

## Contact

For questions or issues:
- Open an issue in this repository
- See the paper for author contact information
