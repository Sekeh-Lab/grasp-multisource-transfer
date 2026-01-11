# Baseline Training

Training code for temporal distribution shift experiments across three benchmark datasets.

## Overview

We provide complete training pipelines for three temporal shift datasets:
- **CLEAR-10**: 10-class continual learning (5 two-year bins)
- **CLEAR-100**: 30-class continual learning (5 two-year bins)
- **Yearbook**: Binary decade classification (4 temporal periods)

Each dataset has its own subdirectory with a complete, self-contained training pipeline.

**Note on CLEAR datasets:** After preprocessing, single-year subsets (year_1, year_2, etc.) were removed to improve training efficiency. Only merged 2-year bins (year_1-2, year_3-4, year_5-6, year_7-8, year_9-10) are available for training.

## Directory Structure

```
baseline_training/
├── clear10/                    # CLEAR-10 (10 classes)
│   ├── model_clear10.py        # PyTorch Lightning model
│   ├── train_clear10.py        # Training script
│   ├── data_module_clear10.py  # Data loading module
│   ├── train_all_subsets.py    # Train all temporal bins
│   └── config_clear10.yaml     # Configuration file
│
├── clear100_30classes/         # CLEAR-100 (30 classes)
│   ├── model_clear100.py
│   ├── train_clear100.py
│   ├── data_module_clear100.py
│   ├── train_all_subsets.py
│   └── config_clear100.yaml
│
└── yearbook/                   # Yearbook (binary)
    ├── model_yearbook.py
    ├── train_yearbook.py
    ├── data_module_yearbook.py
    ├── train_all_subsets.py
    └── config_yearbook.yaml
```

## Supported Models

Four architectures selected for continual learning research:

- **mobilevit-xxs** - 1.3M parameters, transformer, LR=3e-4, batch=32
- **mobilevit-xs** - 2.3M parameters, transformer, LR=3e-4, batch=32
- **resnet-50** - 25.6M parameters, CNN, LR=1e-4, batch=32
- **efficientnet-b1** - 7.8M parameters, CNN, LR=2e-4, batch=32

## Requirements

```bash
# Core dependencies
pip install torch torchvision pytorch-lightning
pip install transformers datasets
pip install matplotlib seaborn scikit-learn
pip install pyyaml psutil
```

### Dataset Setup

Datasets must be preprocessed first. Run the preprocessing scripts (see `../preprocessing/` directory) to organize data into the expected structure:

```
datasets/
├── CLEAR10/
│   ├── year_1-2/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── year_3-4/
│   └── ... (5 two-year bins total)
│
├── CLEAR100_30classes/
│   ├── year_1-2/
│   └── ... (5 two-year bins total)
│
└── Yearbook_Decades/
    ├── before_1950s/
    ├── 1950s_1960s/
    ├── 1970s_1980s/
    └── 1990s_and_later/
```

**Important**: Individual year subsets (year_1, year_2, etc.) are not available - they were removed after preprocessing to improve efficiency.

## Dataset Details

### CLEAR-10 Classes

CLEAR-10 originally contains 11 classes, but **BACKGROUND** is deliberately excluded from training:

**Used in training (10 classes):**
```
baseball, bus, camera, cosplay, dress,
hockey, laptop, racing, soccer, sweater
```

**Excluded:**
- BACKGROUND (11th class - deliberately ignored)

### CLEAR-100 Classes

CLEAR-100 uses a 30-class subset selected for stronger temporal patterns:
```
airplane, aquarium, baseball, beer, boat, bookstore, bowling_ball, 
bridge, bus, camera, castle, chocolate, coins, diving, field_hockey, 
food_truck, football, guitar, hair_salon, helicopter, horse_riding, 
motorcycle, pet_store, racing_car, shopping_mall, skyscraper, soccer, 
stadium, train, video_game
```

### Yearbook Subsets

Yearbook uses 4 binary classification tasks, each comparing consecutive decade pairs:
- **before_1950s**: 1930s vs 1940s
- **1950s_1960s**: 1950s vs 1960s
- **1970s_1980s**: 1970s vs 1980s
- **1990s_and_later**: 1990s vs 2000s+

### Temporal Bins

Both CLEAR datasets use merged 2-year bins for efficiency:
- **Available**: year_1-2, year_3-4, year_5-6, year_7-8, year_9-10
- **Not available**: year_1, year_2, ..., year_10 (removed after preprocessing)

The merged bins provide improved temporal coverage while reducing training time and storage requirements.

## Quick Start

### CLEAR-10

Train a single temporal bin:
```bash
cd baseline_training/clear10
python train_clear10.py --year_name year_1-2
```

Train all 5 temporal bins:
```bash
python train_all_subsets.py
```

Using configuration file:
```bash
python train_clear10.py --config config_clear10.yaml
```

**Available temporal bins**: year_1-2, year_3-4, year_5-6, year_7-8, year_9-10

### CLEAR-100

Train a single temporal bin:
```bash
cd baseline_training/clear100_30classes
python train_clear100.py --year_name year_1-2
```

Train all 5 temporal bins:
```bash
python train_all_subsets.py
```

**Available temporal bins**: year_1-2, year_3-4, year_5-6, year_7-8, year_9-10

### Yearbook

Train a single subset:
```bash
cd baseline_training/yearbook
python train_yearbook.py --subset_name before_1950s
```

Train all 4 subsets:
```bash
python train_all_subsets.py
```

**Available subsets**: before_1950s, 1950s_1960s, 1970s_1980s, 1990s_and_later

## Training Options

### Common Arguments

```bash
# Model selection
python train_clear10.py --model_name mobilevit-xxs

# Training parameters
python train_clear10.py \
    --batch_size 32 \
    --max_epochs 15 \
    --learning_rate 3e-4 \
    --num_workers 4

# Hardware
python train_clear10.py \
    --accelerator auto \
    --devices 1 \
    --precision 16-mixed

# Paths
python train_clear10.py \
    --data_root ../../datasets/CLEAR10 \
    --checkpoint_dir ./model_checkpoints \
    --log_dir ./logs
```

### Dataset-Specific Arguments

**CLEAR-10/CLEAR-100:**
```bash
--year_name year_1-2                 # Temporal bin (merged 2-year bins only)
```

**Yearbook:**
```bash
--subset_name before_1950s           
```

## Configuration Files

Each dataset has a YAML configuration file with recommended settings:

```yaml
# Example: config_clear10.yaml
data_root: "../../datasets/CLEAR10"
year_name: "year_1-2"  # Options: year_1-2, year_3-4, year_5-6, year_7-8, year_9-10
model_name: "mobilevit-xxs"
batch_size: 32
max_epochs: 15
learning_rate: 3.0e-4
seed: null  # Random seed for experiments
```

Edit the config file and run:
```bash
python train_clear10.py --config config_clear10.yaml
```

Command-line arguments override config file settings.

**Note**: Only merged 2-year bins are available for CLEAR datasets (year_1-2 through year_9-10).

## Training All Temporal Bins

Each dataset includes `train_all_subsets.py` for sequential training:

```bash
# Train all bins with default settings
python train_all_subsets.py

# Customize training
python train_all_subsets.py \
    --model_name resnet-50 \
    --max_epochs 20 \
    --batch_size 64 \
    --learning_rate 1e-4
```

## Output Files

### Checkpoints

Saved to `model_checkpoints/` with automatic naming:
```
{model}_{subset}-best-acc-epoch-{XX}-acc-{YY.YY}.ckpt
```

Examples:
```
mobilevit-xxs_year_1-2-best-acc-epoch-12-acc-87.45.ckpt
resnet-50_before_1950s-best-acc-epoch-08-acc-91.23.ckpt
```

### Logs

- **TensorBoard**: `logs/tensorboard/{experiment_name}/`
- **CSV**: `logs/csv/{experiment_name}/`
- **Plots**: `logs/plots/` (confusion matrices)

View with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

### Results Files

Training scripts save metrics to:
```
model_checkpoints/{model}_{subset}-best-acc-*.txt
```

View all results:
```bash
python train_clear10.py --view_results
```

## Advanced Usage

### Multi-GPU Training

```bash
# Use 2 GPUs
python train_clear10.py --devices 2

# Use specific GPUs
python train_clear10.py --devices "0,1"
```

### Mixed Precision

```bash
# 16-bit mixed precision (faster, less memory)
python train_clear10.py --precision 16-mixed

# BF16 (on supported hardware)
python train_clear10.py --precision bf16-mixed
```

### Resume Training

```bash
python train_clear10.py \
    --resume_from_checkpoint model_checkpoints/mobilevit-xxs_year_1-2-best-acc-epoch-10.ckpt
```

## Model Architecture Details

All models use:
- **Input size**: 224×224
- **Preprocessing**: ImageNet normalization
- **Optimizer**: AdamW with differential learning rates
  - Backbone: base learning rate
  - Classifier: 10× base learning rate
- **Scheduler**: Cosine decay with linear warmup (2 epochs)
- **Regularization**: Weight decay (0.01), gradient clipping (1.0)

## Frequently Asked Questions

### Why are individual years (year_1, year_2, etc.) not available?

Individual year subsets were removed after preprocessing to improve efficiency:
- **Storage**: Reduces dataset size significantly
- **Training time**: Fewer temporal bins to train (5 instead of 10-11)
- **Research focus**: Merged 2-year bins provide sufficient temporal coverage and samples for continual learning experiments

The merged bins (year_1-2, year_3-4, year_5-6, year_7-8, year_9-10) balance temporal granularity with practical efficiency.

### Why does CLEAR-10 exclude the BACKGROUND class?

CLEAR-10 originally has 11 classes, but we exclude BACKGROUND for research purposes:
- **Temporal signal**: BACKGROUND class dilutes temporal shift patterns
- **Standard practice**: Continual learning research typically excludes generic/background classes
- **Focus**: We focus on the 10 object classes with clear temporal evolution

Training uses only: baseball, bus, camera, cosplay, dress, hockey, laptop, racing, soccer, sweater.

### Can I add back individual years or BACKGROUND class?

The preprocessing scripts can be modified to include these, but:
- Current experiments and baselines assume merged bins only
- Checkpoints are named and organized for merged bins
- Adding individual years requires re-running all preprocessing

For custom experiments, modify the preprocessing scripts in `../preprocessing/` directory.

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train_clear10.py --batch_size 16

# Use gradient accumulation
python train_clear10.py --accumulate_grad_batches 2

# Use 16-bit precision
python train_clear10.py --precision 16-mixed
```

### Slow Data Loading

```bash
# Reduce workers
python train_clear10.py --num_workers 2

# Or disable workers
python train_clear10.py --num_workers 0
```

### CUDA Out of Memory with Workers

```bash
# Set num_workers to 0
python train_clear10.py --num_workers 0
```

### "Year directory not found" (CLEAR)

- Check that you've run the preprocessing scripts
- Verify the `data_root` path is correct
- Remember: only merged bins (year_1-2, etc.) are available, not individual years

### "Subset directory not found" (Yearbook)

- Check that you've run `yearbook_preprocessing.py`
- Verify the `data_root` path points to `Yearbook_Decades/`
- Check that the subset name matches exactly: before_1950s, 1950s_1960s, 1970s_1980s, or 1990s_and_later

## File Descriptions

### `model_*.py`

PyTorch Lightning module containing:
- Model architecture loading (HuggingFace or torchvision)
- Training/validation/test step logic
- Metrics tracking (accuracy, F1, precision, recall, AUROC)
- Optimizer configuration with differential learning rates
- Checkpoint save/load functionality

### `train_*.py`

Main training script with:
- Argument parsing and configuration loading
- Data module initialization
- Trainer setup with callbacks
- Metrics logging and visualization
- Results viewing mode

### `data_module_*.py`

PyTorch Lightning DataModule with:
- Dataset loading from preprocessed splits
- Image preprocessing and augmentation
- DataLoader configuration
- Automatic model-specific preprocessing detection

### `train_all_subsets.py`

Wrapper script that:
- Sequentially trains all temporal bins/subsets
- Tracks success/failure for each
- Reports comprehensive summary
- Handles interruptions gracefully

### `config_*.yaml`

Configuration file with:
- Model selection
- Training hyperparameters
- Hardware settings
- Logging configuration
- Recommended settings per model

## Integration with Experiments

The baseline training code is used by the experiments in `../experiments/`:

```python
# Example: Loading a trained baseline model
from baseline_training.clear10.model_clear10 import CLEAR10Classifier

model = CLEAR10Classifier.load_from_checkpoint(
    "model_checkpoints/mobilevit-xxs_year_1-2-best-acc.ckpt"
)
```

See `../experiments/shared/model_utils.py` for helper functions.

## Citation

Dataset citations:
- **CLEAR**: Lin et al., "CLEAR: A Dataset for Continual LEArning on Real-Robot Sensory Data", NeurIPS 2021
- **Yearbook**: Ginosar et al., "A Century of Portraits: A Visual Historical Record of American High School Yearbooks", ICCV Workshop 2015

## License

Training code: Released for research purposes only.

Datasets: Please refer to original sources:
- CLEAR: https://clear-benchmark.github.io/
- Yearbook: https://shiry.ttic.edu/projects/yearbooks/yearbooks.html
