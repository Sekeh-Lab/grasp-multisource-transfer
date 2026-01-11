#!/usr/bin/env python3
"""
Yearbook preprocessing: organize into train/val/test splits by temporal periods.
Binary decade classification within each period (e.g., 1930s vs 1940s for before_1950s).
"""
import os
import shutil
import re
import random
from collections import defaultdict

SOURCE_DIR = "./faces_aligned_small_mirrored_co_aligned_cropped_cleaned"
OUTPUT_DIR = "./Yearbook_Decades"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def get_period_name(year):
    """Map year to temporal period."""
    if year < 1950:
        return "before_1950s"
    elif 1950 <= year < 1970:
        return "1950s_1960s"
    elif 1970 <= year < 1990:
        return "1970s_1980s"
    else:
        return "1990s_and_later"


def get_decade_class(year, period):
    """
    Map year to decade class within its period (binary classification).
    
    before_1950s: 1930s (anything before 1940, including 1905-1939), 1940s (1940-1949)
    1950s_1960s: 1950s (1950-1959), 1960s (1960-1969)
    1970s_1980s: 1970s (1970-1979), 1980s (1980-1989)
    1990s_and_later: 1990s (1990-1999), 2000s (2000-2013)
    """
    if period == "before_1950s":
        return "1940s" if year >= 1940 else "1930s"
    elif period == "1950s_1960s":
        return "1960s" if year >= 1960 else "1950s"
    elif period == "1970s_1980s":
        return "1980s" if year >= 1980 else "1970s"
    else:  # 1990s_and_later
        return "2000s" if year >= 2000 else "1990s"


def extract_year_from_filename(filename):
    """Extract year from filename like '1905_Ohio_Cleveland_Central_0-1.png'."""
    match = re.match(r'(\d{4})_', filename)
    if match:
        return int(match.group(1))
    return None


def collect_files_by_period():
    """Scan directories and collect files by period and decade class."""
    files_by_period = defaultdict(lambda: {'1930s': [], '1940s': [], '1950s': [], 
                                            '1960s': [], '1970s': [], '1980s': [],
                                            '1990s': [], '2000s': []})
    
    print("Scanning files...")
    
    # Scan both M and F directories
    for gender in ['M', 'F']:
        gender_dir = os.path.join(SOURCE_DIR, gender)
        
        if not os.path.exists(gender_dir):
            print(f"[WARNING] Directory {gender_dir} not found")
            continue
        
        image_files = [f for f in os.listdir(gender_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} files from {gender} directory...")
        
        for idx, filename in enumerate(image_files):
            year = extract_year_from_filename(filename)
            
            if year is None:
                continue
            
            period_name = get_period_name(year)
            decade_class = get_decade_class(year, period_name)
            full_path = os.path.join(gender_dir, filename)
            files_by_period[period_name][decade_class].append((full_path, filename, year))
            
            if (idx + 1) % 5000 == 0:
                print(f"Scanned {idx + 1}/{len(image_files)} files")
    
    return files_by_period


def stratified_split(files_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split files by decade class maintaining train/val/test ratios."""
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    for decade_class in files_dict.keys():
        if not files_dict[decade_class]:
            continue
            
        files = files_dict[decade_class].copy()
        random.shuffle(files)
        
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits['train'][decade_class] = files[:train_end]
        splits['val'][decade_class] = files[train_end:val_end]
        splits['test'][decade_class] = files[val_end:]
    
    return splits


def copy_files_and_collect_stats(splits, period_name, output_base):
    """Copy files to train/val/test directories and collect stats."""
    stats = defaultdict(lambda: {
        'by_year': defaultdict(int),
        'by_decade': defaultdict(int),
        'total': 0
    })
    
    for split_name in ['train', 'val', 'test']:
        for decade_class in splits[split_name].keys():
            dest_dir = os.path.join(output_base, period_name, split_name, decade_class)
            os.makedirs(dest_dir, exist_ok=True)
            
            files = splits[split_name][decade_class]
            
            for idx, (src_path, filename, year) in enumerate(files):
                dest_path = os.path.join(dest_dir, filename)
                try:
                    shutil.copy2(src_path, dest_path)
                    stats[split_name]['by_year'][year] += 1
                    stats[split_name]['by_decade'][decade_class] += 1
                    stats[split_name]['total'] += 1
                except Exception as e:
                    print(f"[WARNING] Copy failed: {filename}")
                
                if (idx + 1) % 5000 == 0:
                    print(f"    {split_name}/{decade_class}: {idx + 1}/{len(files)}")
    
    return stats


def organize_dataset():
    """Main preprocessing pipeline."""
    print("Yearbook Preprocessing")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Task: Binary decade classification within each period")
    print(f"Periods: 4 (before_1950s, 1950s_1960s, 1970s_1980s, 1990s_and_later)")
    print(f"Split: 80/10/10 train/val/test (seed={RANDOM_SEED})\n")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files_by_period = collect_files_by_period()
    
    if not files_by_period:
        print("[ERROR] No files found")
        return None
    
    all_stats = {}
    period_order = ["before_1950s", "1950s_1960s", "1970s_1980s", "1990s_and_later"]
    decade_map = {
        "before_1950s": ["1930s", "1940s"],
        "1950s_1960s": ["1950s", "1960s"],
        "1970s_1980s": ["1970s", "1980s"],
        "1990s_and_later": ["1990s", "2000s"]
    }
    
    print("Creating splits")
    
    for period_name in period_order:
        if period_name not in files_by_period:
            print(f"\n{period_name}: No files")
            continue
        
        print(f"\n{period_name}:")
        
        files = files_by_period[period_name]
        decades = decade_map[period_name]
        
        # Calculate totals
        total = sum(len(files.get(d, [])) for d in decades)
        
        if total == 0:
            print("No images")
            continue
        
        print(f"Total: {total} images")
        for decade in decades:
            count = len(files.get(decade, []))
            print(f"  {decade}: {count}")
        
        splits = stratified_split(files)
        
        print(f"Copying files...")
        stats = copy_files_and_collect_stats(splits, period_name, OUTPUT_DIR)
        all_stats[period_name] = stats
        
        print(f"Split results:")
        for split in ['train', 'val', 'test']:
            total_split = stats[split]['total']
            split_pct = (total_split / total) * 100 if total > 0 else 0
            print(f"{split}: {total_split} ({split_pct:.1f}%)")
            for decade in decades:
                count = stats[split]['by_decade'].get(decade, 0)
                print(f"{decade}: {count}")
    
    return all_stats, period_order, decade_map


def print_summary(all_stats, period_order, decade_map):
    """Print overall summary statistics."""
    print("\nSummary Statistics")
    
    grand_total = 0
    total_by_split = {'train': 0, 'val': 0, 'test': 0}
    
    for period in period_order:
        if period not in all_stats:
            continue
        
        period_total = 0
        for split in ['train', 'val', 'test']:
            if split in all_stats[period]:
                period_total += all_stats[period][split]['total']
                total_by_split[split] += all_stats[period][split]['total']
        
        grand_total += period_total
        print(f"{period}: {period_total} images")
    
    print(f"\nTotal: {grand_total} images")
    print(f"\nBy split:")
    for split in ['train', 'val', 'test']:
        total = total_by_split[split]
        pct = (total / grand_total) * 100 if grand_total > 0 else 0
        print(f"{split}: {total} ({pct:.1f}%)")


def main():
    """Run preprocessing."""
    result = organize_dataset()
    
    if result:
        all_stats, period_order, decade_map = result
        print_summary(all_stats, period_order, decade_map)
        
        print("\nPreprocessing complete!")
        print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
