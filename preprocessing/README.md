# Dataset Preprocessing Scripts

Python scripts to organize raw benchmark datasets into train/val/test splits for temporal distribution shift experiments.

## Available Scripts

Three preprocessing scripts for the datasets used in this paper:

- `clear10_preprocessing.py` - CLEAR-10 (10 classes, 5 two-year bins)
- `clear100_preprocessing.py` - CLEAR-100 (30 selected classes, 5 two-year bins)  
- `yearbook_preprocessing.py` - Yearbook portraits (binary decade classification, 4 periods)

All scripts use fixed random seed (42) for reproducible splits and require only Python standard library.

## Quick Start

Download the raw dataset, edit the paths in the script, then run it:

```bash
# Example for CLEAR-10
python clear10_preprocessing.py
```

The script will create organized train/val/test directories with 80/10/10 splits.

## Requirements

- Python 3.8+
- No external packages needed (uses only standard library)

## Dataset Downloads

### CLEAR-10

**Source**: https://clear-benchmark.github.io/

Download the CLEAR-10 benchmark (10 object classes across temporal buckets). After downloading, extract the archive.

Expected directory structure:
```
clear-10/
└── train_image_only/
    └── labeled_images/
        ├── 1/
        │   ├── baseball/
        │   ├── bus/
        │   ├── camera/
        │   └── ... (10 classes total)
        ├── 2/
        └── ... (10 year buckets)
```

Update these lines in `clear10_preprocessing.py`:
```python
SOURCE_DIR = "./clear-10/train_image_only"  # Path to downloaded data
OUTPUT_DIR = "./clear-10/CLEAR10"            # Where to save processed data
```

Run: `python clear10_preprocessing.py`

Output structure:
```
CLEAR10/
├── year_1/
│   ├── train/baseball/
│   ├── val/baseball/
│   └── test/baseball/
├── year_2/
├── ...
├── year_10/
├── year_1-2/  (merged 2-year bins)
├── year_3-4/
└── ...
```

### CLEAR-100

**Source**: https://clear-benchmark.github.io/

Download the CLEAR-100 benchmark (100 classes, we use a 30-class subset). Extract after downloading.

Expected directory structure:
```
clear-100/
└── train_image_only/
    └── labeled_images/
        ├── 1/
        │   ├── airplane/
        │   ├── aquarium/
        │   └── ... (100 classes available)
        └── ... (11 year buckets)
```

The script automatically filters to these 30 classes:
```
airplane, aquarium, baseball, beer, boat, bookstore, bowling_ball, 
bridge, bus, camera, castle, chocolate, coins, diving, field_hockey, 
food_truck, football, guitar, hair_salon, helicopter, horse_riding, 
motorcycle, pet_store, racing_car, shopping_mall, skyscraper, soccer, 
stadium, train, video_game
```

Update these lines in `clear100_preprocessing.py`:
```python
SOURCE_DIR = "./clear-100/train_image_only"
OUTPUT_DIR = "./clear-100/CLEAR100_30classes"
```

Run: `python clear100_preprocessing.py`

Output: Same structure as CLEAR-10 but with 30 classes and `CLEAR100_30classes/` as the root.

### Yearbook
**Source**: https://shiry.ttic.edu/projects/yearbooks/yearbooks.html

Download `faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar` (the aligned, cleaned version). Extract it.

Expected directory structure:
```
faces_aligned_small_mirrored_co_aligned_cropped_cleaned/
├── M/
│   ├── 1905_Ohio_Cleveland_Central_0-1.png
│   ├── 1906_Ohio_Cleveland_Central_0-2.png
│   └── ... (male portraits ~1905-2013)
└── F/
    ├── 1905_Ohio_Cleveland_Central_0-1.png
    └── ... (female portraits ~1905-2013)
```

The script organizes images into 4 temporal periods based on year extracted from filename:
- `before_1950s` (1905-1949)
- `1950s_1960s` (1950-1969)
- `1970s_1980s` (1970-1989)
- `1990s_and_later` (1990-2013)

**Binary decade classification** within each temporal period. The script extracts the year from each filename and assigns it to:

1. A **period** (4 total spanning 108 years)
2. A **decade class** within that period (binary: 0 or 1)

Update these lines in `yearbook_preprocessing.py`:
```python
SOURCE_DIR = "./faces_aligned_small_mirrored_co_aligned_cropped_cleaned"
OUTPUT_DIR = "./Yearbook_Decades"
```

Run: `python yearbook_preprocessing.py`

Output structure:
```
Yearbook_Decades/
  before_1950s/
    train/
      1930s/  (contains years 1905-1939)
      1940s/  (contains years 1940-1949)
    val/
      1930s/
      1940s/
    test/
      1930s/
      1940s/

  1950s_1960s/
    train/
      1950s/  (contains years 1950-1959)
      1960s/  (contains years 1960-1969)
    val/
    test/

  1970s_1980s/
    train/
      1970s/  (contains years 1970-1979)
      1980s/  (contains years 1980-1989)
    val/
    test/

  1990s_and_later/
    train/
      1990s/  (contains years 1990-1999)
      2000s/  (contains years 2000-2013)
    val/
    test/
```

## Split Details

All scripts create stratified 80/10/10 train/val/test splits:
- **CLEAR-10 & CLEAR-100**: Stratified by class (ensures each class has same ratio)
- **Yearbook**: Stratified by gender (ensures M/F balance)

## Troubleshooting

**"Source directory not found"**
- Check that SOURCE_DIR points to the correct extracted dataset location
- For CLEAR datasets, make sure path includes `/train_image_only` if that's where your data is
- Use absolute paths if relative paths aren't working

**"No year directories found" (CLEAR)**
- Verify the dataset extracted correctly
- Check that `labeled_images/` subdirectory exists
- Make sure you downloaded the correct dataset variant

**"No files found" (Yearbook)**
- Verify M/ and F/ directories exist in the source folder
- Check that filenames follow the pattern: `YYYY_...png` or `YYYY_...jpg`
- The year must be in the first 4 characters of each filename

**Script runs but creates empty directories**
- Check file permissions on source directories
- Verify image files have correct extensions (.jpg, .jpeg, .png, .gif, .bmp)
- For CLEAR-100, note that only 30 classes are processed (by design)

## Citation

If you use these preprocessing scripts or the organized datasets:

```bibtex
@inproceedings{grasp2026,
  title={GRASP: Gradient-Aligned Sequential Parameter Transfer for Memory-Efficient Multi-Source Learning},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2026}
}
```

Dataset citations:
- **CLEAR**: Lin et al., "CLEAR: A Dataset for Continual LEArning on Real-Robot Sensory Data", NeurIPS 2021
- **Yearbook**: Ginosar et al., "A Century of Portraits: A Visual Historical Record of American High School Yearbooks", ICCV Workshop 2015

## License

Preprocessing scripts: MIT License

Datasets: Please refer to original datasets:
- CLEAR: https://clear-benchmark.github.io/
- Yearbook: https://shiry.ttic.edu/projects/yearbooks/yearbooks.html
