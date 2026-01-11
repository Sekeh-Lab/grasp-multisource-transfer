#!/usr/bin/env python3
"""
CLEAR-100 (30-class subset) preprocessing: organize into train/val/test splits with 2-year bins.
"""
import os
import shutil
import json
import random
from collections import defaultdict

SOURCE_DIR = "./clear-100/train_image_only"
OUTPUT_DIR = "./clear-100/CLEAR100_30classes"
RANDOM_SEED = 42

# Selected 30 classes for experiments
SELECTED_CLASSES = [
    'airplane', 'aquarium', 'baseball', 'beer', 'boat', 'bookstore',
    'bowling_ball', 'bridge', 'bus', 'camera', 'castle', 'chocolate',
    'coins', 'diving', 'field_hockey', 'food_truck', 'football', 'guitar',
    'hair_salon', 'helicopter', 'horse_riding', 'motorcycle', 'pet_store',
    'racing_car', 'shopping_mall', 'skyscraper', 'soccer', 'stadium',
    'train', 'video_game'
]

random.seed(RANDOM_SEED)


def load_class_names(source_dir):
    """Load and filter to selected 30 classes."""
    possible_paths = [
        os.path.join(source_dir, 'class_names.txt'),
        os.path.join(source_dir, 'train_image_only', 'class_names.txt'),
        os.path.join(source_dir, 'test', 'class_names.txt')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_classes = [line.strip() for line in f if line.strip()]
            
            # Filter to selected classes only
            filtered = [c for c in all_classes if c in SELECTED_CLASSES]
            
            if len(filtered) != len(SELECTED_CLASSES):
                print(f"[WARNING] Found {len(filtered)}/{len(SELECTED_CLASSES)} selected classes")
            
            return filtered
    
    print("[ERROR] class_names.txt not found")
    return []


def find_dataset_paths(source_dir):
    """Find paths for images and metadata directories."""
    possible_bases = [
        os.path.join(source_dir, 'train_image_only'),
        os.path.join(source_dir, 'test'),
        source_dir
    ]
    
    for base in possible_bases:
        images_base = os.path.join(base, 'labeled_images')
        metadata_base = os.path.join(base, 'labeled_metadata')
        
        if os.path.exists(images_base) and os.path.exists(metadata_base):
            return images_base, metadata_base, base
    
    return None, None, None


def collect_files_by_year(source_dir, classes):
    """Scan directories and collect files by year and class."""
    files_by_year = defaultdict(lambda: defaultdict(list))
    metadata_by_year = defaultdict(dict)
    
    print("Scanning files...")
    
    images_base, metadata_base, base_dir = find_dataset_paths(source_dir)
    
    if not images_base or not metadata_base:
        print("[ERROR] Could not find labeled_images and labeled_metadata directories")
        return None, None, None, None
    
    print(f"Images: {images_base}")
    print(f"Metadata: {metadata_base}\n")
    
    if not os.path.exists(images_base):
        print(f"[ERROR] Images directory does not exist: {images_base}")
        return None, None, None, None
    
    year_dirs = [d for d in os.listdir(images_base) 
                 if os.path.isdir(os.path.join(images_base, d)) and d.isdigit()]
    years = sorted(year_dirs, key=int)
    
    if not years:
        print("[ERROR] No year directories found")
        return None, None, None, None
    
    print(f"Found {len(years)} years: {years}\n")
    
    for year in years:
        print(f"Year {year}:")
        year_image_dir = os.path.join(images_base, year)
        year_metadata_dir = os.path.join(metadata_base, year)
        
        for class_name in classes:
            class_image_dir = os.path.join(year_image_dir, class_name)
            class_metadata_file = os.path.join(year_metadata_dir, f"{class_name}.json")
            
            if os.path.exists(class_metadata_file):
                metadata_by_year[year][class_name] = class_metadata_file
            
            if not os.path.exists(class_image_dir):
                continue
            
            try:
                image_files = [f for f in os.listdir(class_image_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                
                for img_file in image_files:
                    img_path = os.path.join(class_image_dir, img_file)
                    files_by_year[year][class_name].append((img_path, img_file, class_metadata_file))
                
                if len(image_files) > 0:
                    print(f"  {class_name}: {len(image_files)} images")
                
            except Exception as e:
                print(f"  {class_name}: scan error - {e}")
        
        print()
    
    return files_by_year, metadata_by_year, years, classes


def stratified_split(files_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split files by class maintaining train/val/test ratios."""
    splits = {'train': defaultdict(list), 
              'val': defaultdict(list), 
              'test': defaultdict(list)}
    
    for class_name, files in files_dict.items():
        if not files:
            continue
        
        files_copy = files.copy()
        random.shuffle(files_copy)
        
        total = len(files_copy)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits['train'][class_name] = files_copy[:train_end]
        splits['val'][class_name] = files_copy[train_end:val_end]
        splits['test'][class_name] = files_copy[val_end:]
    
    return splits


def copy_files_and_collect_stats(splits, year_name, output_base, classes, metadata_sources):
    """Copy files to train/val/test directories and collect stats."""
    stats = defaultdict(lambda: {'by_class': defaultdict(int), 'total': 0})
    
    dest_metadata_dir = os.path.join(output_base, year_name, 'metadata')
    os.makedirs(dest_metadata_dir, exist_ok=True)
    
    # Copy metadata files
    metadata_copied = set()
    for class_name in classes:
        if class_name in metadata_sources:
            src_metadata_path = metadata_sources[class_name]
            if os.path.exists(src_metadata_path) and class_name not in metadata_copied:
                dest_metadata_path = os.path.join(dest_metadata_dir, f"{class_name}.json")
                try:
                    shutil.copy2(src_metadata_path, dest_metadata_path)
                    metadata_copied.add(class_name)
                except Exception as e:
                    print(f"[WARNING] Metadata copy failed for {class_name}: {e}")
    
    for split_name in ['train', 'val', 'test']:
        class_files = splits[split_name]
        
        for class_name in classes:
            if class_name not in class_files or not class_files[class_name]:
                continue
            
            dest_img_dir = os.path.join(output_base, year_name, split_name, class_name)
            os.makedirs(dest_img_dir, exist_ok=True)
            
            files = class_files[class_name]
            for idx, (src_img_path, img_filename, src_metadata_path) in enumerate(files):
                dest_img_path = os.path.join(dest_img_dir, img_filename)
                try:
                    shutil.copy2(src_img_path, dest_img_path)
                    stats[split_name]['by_class'][class_name] += 1
                    stats[split_name]['total'] += 1
                except Exception as e:
                    print(f"[WARNING] Copy failed: {img_filename}")
                
                if (idx + 1) % 5000 == 0:
                    print(f"    {split_name}: {idx + 1}/{len(files)} {class_name}")
    
    return stats


def organize_dataset():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("CLEAR-100 (30-class subset) Preprocessing")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Selected classes: {len(SELECTED_CLASSES)}")
    print(f"Split: 80/10/10 train/val/test (seed={RANDOM_SEED})\n")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classes = load_class_names(SOURCE_DIR)
    if not classes:
        print("[ERROR] No classes found")
        return None
    
    print(f"Using {len(classes)} classes\n")
    
    result = collect_files_by_year(SOURCE_DIR, classes)
    
    if result[0] is None:
        print("[ERROR] No files found")
        return None
    
    files_by_year, metadata_by_year, years, classes = result
    
    all_stats = {}
    
    print("Creating splits")
    
    for year in years:
        if year not in files_by_year:
            print(f"\nYear {year}: No files")
            continue
        
        year_name = f"year_{year}"
        print(f"\n{year_name}:")
        
        files = files_by_year[year]
        total = sum(len(files[cls]) for cls in classes if cls in files)
        
        if total == 0:
            print("No images")
            continue
        
        print(f"Total: {total} images")
        
        splits = stratified_split(files)
        
        print(f"Copying files...")
        stats = copy_files_and_collect_stats(splits, year_name, OUTPUT_DIR, classes, metadata_by_year[year])
        all_stats[year] = stats
        
        print(f"Split results:")
        for split in ['train', 'val', 'test']:
            total_split = stats[split]['total']
            split_pct = (total_split / total) * 100 if total > 0 else 0
            print(f"  {split}: {total_split} ({split_pct:.1f}%)")
    
    return all_stats, years, classes


def merge_year_bins(output_dir, years, classes):
    """Merge consecutive year pairs into 2-year bins."""
    print("Creating 2-year bins")
    
    year_pairs = [
        (1, 2, "year_1-2"),
        (3, 4, "year_3-4"),
        (5, 6, "year_5-6"),
        (7, 8, "year_7-8"),
        (9, 10, "year_9-10")
    ]
    
    for year1, year2, merged_name in year_pairs:
        year1_str = str(year1)
        year2_str = str(year2)
        
        year1_dir = os.path.join(output_dir, f"year_{year1}")
        year2_dir = os.path.join(output_dir, f"year_{year2}")
        
        if not os.path.exists(year1_dir) or not os.path.exists(year2_dir):
            print(f"\n[WARNING] Skipping {merged_name}: source years not found")
            continue
        
        print(f"\n{merged_name}: merging year_{year1} + year_{year2}")
        
        merged_dir = os.path.join(output_dir, merged_name)
        os.makedirs(merged_dir, exist_ok=True)
        
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for split in ['train', 'val', 'test']:
            for class_name in classes:
                merged_class_dir = os.path.join(merged_dir, split, class_name)
                os.makedirs(merged_class_dir, exist_ok=True)
            
            for year_str, year_dir in [(year1_str, year1_dir), (year2_str, year2_dir)]:
                split_dir = os.path.join(year_dir, split)
                
                if not os.path.exists(split_dir):
                    continue
                
                for class_name in classes:
                    src_class_dir = os.path.join(split_dir, class_name)
                    dst_class_dir = os.path.join(merged_dir, split, class_name)
                    
                    if not os.path.exists(src_class_dir):
                        continue
                    
                    try:
                        files = [f for f in os.listdir(src_class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                        
                        for filename in files:
                            src_path = os.path.join(src_class_dir, filename)
                            new_filename = f"y{year_str}_{filename}"
                            dst_path = os.path.join(dst_class_dir, new_filename)
                            
                            shutil.copy2(src_path, dst_path)
                            stats[split] += 1
                    
                    except Exception as e:
                        print(f"[WARNING] Copy error for {class_name} from year_{year_str}: {e}")
        
        # Merge metadata
        metadata_dir = os.path.join(merged_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        for class_name in classes:
            merged_metadata = []
            
            for year_str in [year1_str, year2_str]:
                year_metadata_file = os.path.join(output_dir, f"year_{year_str}", 
                                                   'metadata', f"{class_name}.json")
                
                if os.path.exists(year_metadata_file):
                    try:
                        with open(year_metadata_file, 'r') as f:
                            year_meta = json.load(f)
                            if isinstance(year_meta, list):
                                for entry in year_meta:
                                    if 'filename' in entry:
                                        entry['filename'] = f"y{year_str}_{entry['filename']}"
                                    if 'source_year' not in entry:
                                        entry['source_year'] = int(year_str)
                                merged_metadata.extend(year_meta)
                    except Exception as e:
                        print(f"[WARNING] Metadata read error for {class_name} year_{year_str}: {e}")
            
            if merged_metadata:
                merged_metadata_file = os.path.join(metadata_dir, f"{class_name}.json")
                try:
                    with open(merged_metadata_file, 'w') as f:
                        json.dump(merged_metadata, f, indent=2)
                except Exception as e:
                    print(f"[WARNING] Metadata write error for {class_name}: {e}")
        
        print(f"[SUCCESS] Complete: train={stats['train']}, val={stats['val']}, test={stats['test']}")


def main():
    """Run preprocessing."""
    result = organize_dataset()
    
    if result:
        all_stats, years, classes = result
        merge_year_bins(OUTPUT_DIR, years, classes)
        
        print("Preprocessing complete!")
        print(f"\nOutput: {OUTPUT_DIR}")
        print(f"Classes: {len(classes)}")

if __name__ == "__main__":
    main()
