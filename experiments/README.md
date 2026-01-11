# Multi-Source Transfer Learning Experiments

Comprehensive experimental framework for evaluating multi-source transfer learning methods on temporal distribution shift benchmarks.

## Overview

This directory contains implementations of four multi-source transfer learning methods evaluated on three temporal shift datasets:

**Methods:**
- **ENSEMBLE** - Multi-model voting ensemble (soft/hard voting)
- **FUSION** - Parameter averaging with fine-tuning (referred to as "Multi-Source" baseline in the paper)
- **GRASP** - Gradient-aligned sequential parameter transfer (our proposed method)
- **PEARL** - Parameter-efficient adapter-based learning (also one of our methods)

**Note on naming:** The `fusion/` directory implements the **Multi-Source baseline** method described in the paper. We use "fusion" in the code for brevity, as it describes the core operation (fusing/averaging parameters from multiple source models before fine-tuning on the target domain).

**Datasets:**
- **CLEAR-10**: 10-class continual learning (5 temporal bins)
- **CLEAR-100**: 30-class continual learning (5 temporal bins)
- **Yearbook**: Binary decade classification (4 temporal periods)

**Models:**
- mobilevit-xxs (1.3M parameters)
- mobilevit-xs (2.3M parameters)
- resnet-50 (25.6M parameters)
- efficientnet-b1 (7.8M parameters)

## Directory Structure

```
experiments/
├── shared/                      # Shared utilities (all methods)
│   ├── __init__.py
│   ├── data_utils.py           # Dataset loading
│   ├── model_utils.py          # Model loading and utilities
│   ├── flops_counter.py        # FLOPs calculation
│   ├── metrics_utils.py        # Performance metrics
│   └── gpu_memory_tracker.py  # GPU monitoring
│
├── ensemble/                    # Voting ensemble method
│   ├── run_ensemble_experiment.py
│   ├── run_all_ensemble_clear10.py
│   ├── run_all_ensemble_clear100.py
│   └── run_all_ensemble_yearbook.py
│
├── fusion/                      # Parameter averaging method
│   ├── run_fusion_experiment.py
│   ├── run_all_fusion_clear10.py
│   ├── run_all_fusion_clear100.py
│   └── run_all_fusion_yearbook.py
│
├── grasp/                       # Gradient-aligned transfer
│   ├── run_grasp_experiment.py
│   ├── run_all_grasp_clear10.py
│   ├── run_all_grasp_clear100.py
│   └── run_all_grasp_yearbook.py
│
└── pearl/                       # Adapter-based learning
    ├── run_pearl_experiment.py
    ├── run_all_pearl_clear10.py
    ├── run_all_pearl_clear100.py
    └── run_all_pearl_yearbook.py
```

## Prerequisites

### 1. Baseline Models

Train baseline models first using the code in `../baseline_training/`:

```bash
# Train baselines for CLEAR-10
cd ../baseline_training/clear10
python train_all_subsets.py --model_name mobilevit-xxs

# Train baselines for CLEAR-100
cd ../baseline_training/clear100_30classes
python train_all_subsets.py --model_name mobilevit-xxs

# Train baselines for Yearbook
cd ../baseline_training/yearbook
python train_all_subsets.py --model_name mobilevit-xxs
```

This creates checkpoint files needed by the experiments.

### 2. Dependencies

```bash
pip install torch torchvision pytorch-lightning
pip install transformers datasets
pip install matplotlib seaborn scikit-learn
pip install numpy pyyaml psutil
```

### 3. Dataset Structure

Ensure preprocessed datasets are available:
```
datasets/
├── CLEAR10/
│   ├── year_1-2/
│   ├── year_3-4/
│   ├── year_5-6/
│   ├── year_7-8/
│   └── year_9-10/
├── CLEAR100_30classes/
│   └── (same structure)
└── Yearbook_Decades/
    ├── before_1950s/
    ├── 1950s_1960s/
    ├── 1970s_1980s/
    └── 1990s_and_later/
```

## Quick Start

### ENSEMBLE Method

Combine predictions from multiple source models via soft voting:

```bash
cd experiments/ensemble

# Single experiment
python run_ensemble_experiment.py \
    --dataset clear10 \
    --target year_1-2 \
    --sources year_3-4 year_5-6 year_7-8 year_9-10 \
    --model apple/mobilevit-xx-small \
    --trial 1

# All targets with 3 trials each
python run_all_ensemble_clear10.py --num_trials 3 --model mobilevit-xxs
python run_all_ensemble_clear100.py --num_trials 3 --model mobilevit-xxs
python run_all_ensemble_yearbook.py --num_trials 3 --model mobilevit-xxs
```

### FUSION Method

Average source model parameters, then fine-tune on target:

```bash
cd experiments/fusion

# Single experiment
python run_fusion_experiment.py \
    --dataset clear10 \
    --target year_1-2 \
    --sources year_3-4 year_5-6 year_7-8 year_9-10 \
    --model apple/mobilevit-xx-small \
    --trial 1 \
    --finetune_epochs 3

# All targets with 3 trials each
python run_all_fusion_clear10.py --trial 3 --model mobilevit-xxs
python run_all_fusion_clear100.py --trial 3 --model mobilevit-xxs
python run_all_fusion_yearbook.py --trial 3 --model mobilevit-xxs
```

### GRASP Method

Sequential transfer with gradient alignment:

```bash
cd experiments/grasp

# Single experiment
python run_grasp_experiment.py \
    --dataset clear10 \
    --target year_1-2 \
    --sources year_3-4 year_5-6 year_7-8 year_9-10 \
    --model apple/mobilevit-xx-small \
    --trial 1 \
    --alignment_threshold 0.3

# All targets with 3 trials each
python run_all_grasp_clear10.py --num_trials 3 --model mobilevit-xxs
python run_all_grasp_clear100.py --num_trials 3 --model mobilevit-xxs
python run_all_grasp_yearbook.py --num_trials 3 --model mobilevit-xxs
```

### PEARL Method

Adapter composition with LoRA:

```bash
cd experiments/pearl

# Single experiment
python run_pearl_experiment.py \
    --dataset clear10 \
    --target year_1-2 \
    --sources year_3-4 year_5-6 year_7-8 year_9-10 \
    --model apple/mobilevit-xx-small \
    --trial 1 \
    --adapter_dim 16

# All targets with 3 trials each
python run_all_pearl_clear10.py --num_trials 3 --model mobilevit-xxs
python run_all_pearl_clear100.py --num_trials 3 --model mobilevit-xxs
python run_all_pearl_yearbook.py --num_trials 3 --model mobilevit-xxs
```

## Method Details

### ENSEMBLE

**Algorithm**: Load pre-trained source models and combine predictions via voting.

**Key Parameters**:
```bash
--voting soft              # Voting type: 'soft' (average probs) or 'hard' (majority vote)
--batch_size 32           # Batch size for inference
```

**Characteristics**:
- No additional training required
- Linear scaling with number of sources (memory and compute)
- Naturally handles model diversity

**Output**: Ensemble predictions with combined performance metrics.

---

### FUSION

**Algorithm**: 
1. Average parameters from all source models
2. Fine-tune averaged model on target data

**Key Parameters**:
```bash
--finetune_epochs 3       # Epochs for fine-tuning on target
--learning_rate 5e-5      # Fine-tuning learning rate
--batch_size 32           # Batch size
```

**Characteristics**:
- Requires target domain data for fine-tuning
- Single model at inference (memory efficient)
- Fast convergence due to pre-initialized parameters

**Output**: Fine-tuned merged model with target-domain performance.

---

### GRASP

**Algorithm**:
1. Train initial model on first source
2. For each additional source:
   - Compute gradient alignment scores
   - Transfer aligned parameters
   - Fine-tune on next source
3. Final fine-tuning on target

**Key Parameters**:
```bash
--initial_epochs 3              # Epochs for first source
--finetune_epochs 3             # Epochs for each transfer step
--alignment_threshold 0.3       # Gradient alignment cutoff
--num_alignment_batches 1       # Batches for alignment computation
--learning_rate 5e-5            # Learning rate
```

**Characteristics**:
- Sequential transfer with selective parameter updates
- Gradient-based parameter selection
- Balances forward transfer and catastrophic forgetting

**Output**: Sequentially transferred model optimized for target domain.

---

### PEARL

**Algorithm**:
1. Train LoRA adapters on each source domain
2. Learn composition weights for adapters
3. Apply composed adapter to target

**Key Parameters**:
```bash
--adapter_dim 16                # LoRA adapter dimension
--finetune_epochs 3             # Epochs per adapter
--composition_epochs 3          # Epochs for learning composition
--learning_rate 5e-5            # Learning rate
```

**Characteristics**:
- Parameter-efficient (only trains small adapters)
- Modular composition of source knowledge
- Low memory footprint during training

**Output**: Adapter composition optimized for target domain.

## Configuration Options

### Common Arguments (All Methods)

```bash
# Dataset and experiment setup
--dataset clear10                    # Dataset: clear10, clear100, yearbook
--target year_1-2                    # Target temporal bin
--sources year_3-4 year_5-6          # Source temporal bins
--model apple/mobilevit-xx-small     # Model (HuggingFace format or standard name)
--trial 1                            # Trial number for multiple runs

# Hardware
--device auto                        # Device: 'auto', 'cuda', 'cpu'
--batch_size 32                      # Batch size
--num_workers 12                     # DataLoader workers

# Paths
--results_dir results                # Output directory
--data_root ../../datasets           # Dataset root
```

### Method-Specific Arguments

**ENSEMBLE**:
```bash
--voting soft                        # Voting strategy: 'soft' or 'hard'
```

**FUSION**:
```bash
--finetune_epochs 3                  # Fine-tuning epochs
--learning_rate 5e-5                 # Learning rate
```

**GRASP**:
```bash
--initial_epochs 3                   # Initial training epochs
--finetune_epochs 3                  # Fine-tuning epochs per source
--alignment_threshold 0.3            # Gradient alignment threshold
--num_alignment_batches 1            # Alignment computation batches
--learning_rate 5e-5                 # Learning rate
```

**PEARL**:
```bash
--adapter_dim 16                     # LoRA adapter dimension
--finetune_epochs 3                  # Adapter training epochs
--composition_epochs 3               # Composition learning epochs
--learning_rate 5e-5                 # Learning rate
```

## Temporal Bins and Sources

### CLEAR-10 and CLEAR-100

**Available bins**: year_1-2, year_3-4, year_5-6, year_7-8, year_9-10

**Example**: Train on year_1-2 using other bins as sources:
```bash
--target year_1-2 \
--sources year_3-4 year_5-6 year_7-8 year_9-10
```

**Typical setup**: Leave-one-out (4 sources, 1 target, 5 total experiments)

### Yearbook

**Available periods**: before_1950s, 1950s_1960s, 1970s_1980s, 1990s_and_later

**Example**: Train on before_1950s using other periods as sources:
```bash
--target before_1950s \
--sources 1950s_1960s 1970s_1980s 1990s_and_later
```

**Typical setup**: Leave-one-out (3 sources, 1 target, 4 total experiments)

## Model Names

Models can be specified in multiple formats:

```bash
# HuggingFace format (canonical)
--model apple/mobilevit-xx-small
--model apple/mobilevit-x-small
--model microsoft/resnet-50
--model google/efficientnet-b1

# Short format (automatically converted)
--model mobilevit-xxs
--model mobilevit-xs
--model resnet-50
--model efficientnet-b1
```

The `shared/model_utils.py` handles automatic conversion.

## Output Files

### Results Directory Structure

```
results/
├── clear10/
│   ├── mobilevit-xxs_year_1-2_ensemble_trial1/
│   │   ├── metrics.json                      # Performance metrics
│   │   ├── config.json                       # Experiment configuration
│   │   ├── confusion_matrices/               # Visualization
│   │   │   ├── val_confusion_matrix.png
│   │   │   └── test_confusion_matrix.png
│   │   ├── memory_profile.json               # GPU memory usage
│   │   ├── flops_report.json                 # Computational cost
│   │   └── mobilevit-xxs_year_1-2_ensemble_trial1_comprehensive_report.txt
│   └── ...
├── clear100/
└── yearbook/
```

### Metrics JSON

Each experiment saves comprehensive metrics. The example below is to demonstrate the formatting only, and are not actual experimental results:

```json
{
  "dataset": "clear10",
  "target": "year_1-2",
  "sources": ["year_3-4", "year_5-6", "year_7-8", "year_9-10"],
  "model": "apple/mobilevit-xx-small",
  "method": "ensemble",
  "trial": 1,
  "test_metrics": {
    "accuracy": 0.8745,
    "precision": 0.8721,
    "recall": 0.8698,
    "f1_score": 0.8709
  },
  "total_time_minutes": 2.5,
  "gpu_memory": {
    "peak_gpu_memory_gb": 3.2451,
    "is_cuda": true
  },
  "flops": {
    "total_gflops": 45.23,
    "method": "ENSEMBLE"
  }
}
```

### Summary Reports

Batch runners (`run_all_*` scripts) generate summary files:

```
ensemble_mobilevit-xxs_clear10_summary.txt
fusion_mobilevit-xxs_clear10_summary.txt
grasp_30_mobilevit-xxs_clear10_summary.txt  # 30 = alignment threshold (%)
pearl_mobilevit-xxs_clear10_summary.txt
```

Example summary content:
```
==================================================================================================
ENSEMBLE - CLEAR-10 COMPREHENSIVE RESULTS
==================================================================================================

CONFIGURATION
--------------------------------------------------------------------------------------------------
Dataset:           CLEAR-10 (10-class classification)
Model:             mobilevit-xxs
Method:            ENSEMBLE (Soft Voting)
Trials per Target: 3
Total Targets:     5

RESULTS BY TARGET
==================================================================================================

TARGET: YEAR_1-2
Sources: year_3-4, year_5-6, year_7-8, year_9-10
--------------------------------------------------------------------------------------------------
Trials: 3/3

Test Performance (Mean +/- Std):
  Accuracy:  87.45% +/-  1.23%
  Precision: 0.8721 +/- 0.0134
  Recall:    0.8698 +/- 0.0127
  F1-Score:  0.8709 +/- 0.0129

FLOPs (Mean +/- Std):
  Total GFLOPs:          45.23 +/- 0.12
  Validation Inference:  22.34 GFLOPs
  Test Inference:        22.89 GFLOPs

GPU Memory Usage (Mean +/- Std):
  Peak Allocated:        3.2451 +/- 0.0234 GB
  Min Peak:              3.2187 GB
  Max Peak:              3.2712 GB

Average Time: 2.45 +/- 0.67 minutes

...
```

## Multiple Trials

Run multiple trials to assess variance:

```bash
# 3 trials for all targets
python run_all_ensemble_clear10.py --num_trials 3

# 5 trials for single target
for trial in {1..5}; do
    python run_ensemble_experiment.py \
        --dataset clear10 \
        --target year_1-2 \
        --sources year_3-4 year_5-6 year_7-8 year_9-10 \
        --model mobilevit-xxs \
        --trial $trial
done
```

## Shared Utilities

The `shared/` directory contains common utilities used by all methods:

### data_utils.py

Functions for loading datasets:
```python
from shared.data_utils import get_dataloaders

train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    dataset='clear10',
    subset='year_1-2',
    batch_size=32,
    num_workers=4
)
```

### model_utils.py

Functions for loading baseline models:
```python
from shared.model_utils import load_baseline_model, standardize_model_name

# Load pre-trained baseline
model = load_baseline_model(
    dataset='clear10',
    subset='year_1-2',
    model_name='mobilevit-xxs',
    num_classes=10
)

# Standardize model names
standard_name = standardize_model_name('apple/mobilevit-xx-small')
# Returns: 'mobilevit-xxs'
```

### flops_counter.py

Calculate FLOPs for each method:
```python
from shared.flops_counter import get_ensemble_flops_report

flops_report = get_ensemble_flops_report(
    model=model,
    n_sources=4,
    n_val_samples=1000,
    n_test_samples=2000,
    num_classes=10,
    voting='soft'
)
```

### metrics_utils.py

Performance evaluation:
```python
from shared.metrics_utils import calculate_metrics, save_confusion_matrix

metrics = calculate_metrics(labels, predictions, num_classes=10)
save_confusion_matrix(labels, predictions, save_path='cm.png')
```

### gpu_memory_tracker.py

GPU memory monitoring:
```python
from shared.gpu_memory_tracker import GPUMemoryTracker

tracker = GPUMemoryTracker(device)
tracker.checkpoint('experiment_start')
# ... run experiment ...
tracker.checkpoint('experiment_end')
summary = tracker.get_summary()
```

## Troubleshooting

### Baseline Checkpoints Not Found

**Error**: `FileNotFoundError: Baseline checkpoint not found`

**Solution**:
```bash
# Train missing baselines first
cd ../baseline_training/clear10
python train_all_subsets.py --model_name mobilevit-xxs
```

**Alternative**: Specify custom checkpoint directory:
```bash
python run_ensemble_experiment.py \
    --baseline_dir /path/to/baseline_training \
    ...
```

### Out of Memory

**For ENSEMBLE** (multiple models loaded):
```bash
# Reduce batch size
python run_ensemble_experiment.py --batch_size 16 ...

# Reduce number of sources (experiment-specific)
python run_ensemble_experiment.py --sources year_3-4 year_5-6 ...

# Use CPU (slower but no memory limit)
python run_ensemble_experiment.py --device cpu ...
```

**For FUSION/GRASP/PEARL** (single model):
```bash
# Reduce batch size
python run_fusion_experiment.py --batch_size 16 ...

# Use gradient accumulation (FUSION/GRASP/PEARL)
# Modify training scripts to add accumulate_grad_batches
```

### Slow Experiments

```bash
# Reduce DataLoader workers
python run_ensemble_experiment.py --num_workers 2 ...

# Use smaller model
python run_ensemble_experiment.py --model mobilevit-xxs ...

# Reduce training epochs (FUSION/GRASP/PEARL)
python run_fusion_experiment.py --finetune_epochs 2 ...
```

### "Dataset not found"

**Error**: `Dataset directory not found: ../../datasets/CLEAR10/year_1-2`

**Solution**:
```bash
# Check preprocessing completed
ls ../../datasets/CLEAR10/

# Specify correct path
python run_ensemble_experiment.py --data_root /absolute/path/to/datasets ...
```

### FLOPs Calculation Errors

**Error**: `FLOPs counter not available` or calculation fails

**Impact**: Experiments still run, but FLOPs metrics show as "Not available"

**Common causes**:
- Complex model architectures
- Custom layers not supported
- Dynamic computation graphs

**Solution**: Results are still valid, just missing computational cost metrics.

## Comparing Methods

Run all methods on the same configuration:

```bash
# Set common variables
DATASET=clear10
TARGET=year_1-2
SOURCES="year_3-4 year_5-6 year_7-8 year_9-10"
MODEL=mobilevit-xxs
TRIAL=1

# ENSEMBLE
cd experiments/ensemble
python run_ensemble_experiment.py \
    --dataset $DATASET --target $TARGET --sources $SOURCES \
    --model $MODEL --trial $TRIAL

# FUSION
cd ../fusion
python run_fusion_experiment.py \
    --dataset $DATASET --target $TARGET --sources $SOURCES \
    --model $MODEL --trial $TRIAL

# GRASP
cd ../grasp
python run_grasp_experiment.py \
    --dataset $DATASET --target $TARGET --sources $SOURCES \
    --model $MODEL --trial $TRIAL

# PEARL
cd ../pearl
python run_pearl_experiment.py \
    --dataset $DATASET --target $TARGET --sources $SOURCES \
    --model $MODEL --trial $TRIAL
```

Compare results from `results/{dataset}/` directories.

## Citation
Method references:
- **ENSEMBLE**: Standard ensemble learning
- **FUSION**: Matena & Raffel, "Merging Models with Fisher-Weighted Averaging", NeurIPS 2022
- **GRASP**: This work
- **PEARL**: Created by us

Dataset citations:
- **CLEAR**: Lin et al., NeurIPS 2021
- **Yearbook**: Ginosar et al., ICCV Workshop 2015

## License

Experimental code: Released for research purposes only.

Please refer to original papers and datasets for their respective licenses.
