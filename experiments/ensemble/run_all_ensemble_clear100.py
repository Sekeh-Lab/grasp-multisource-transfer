#!/usr/bin/env python
"""
run_all_ensemble_clear100.py - Run all ENSEMBLE experiments for CLEAR-100

Runs comprehensive ENSEMBLE experiments across CLEAR-100 time periods (30 classes).

Usage:
    python run_all_ensemble_clear100.py --num_trials 3
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Import model utilities
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from model_utils import get_huggingface_model_name, standardize_model_name


# CLEAR-100 time periods (joint years only)
ALL_YEARS = ["year_1-2", "year_3-4", "year_5-6", "year_7-8", "year_9-10"]
DATASET = "clear100"


def run_single_experiment(target: str, sources: List[str], model: str, trial: int, args) -> Dict:
    """Run a single ensemble experiment."""
    model_full = get_huggingface_model_name(model)
    
    print(f"\n{'='*80}")
    print(f"Running: Target={target}, Model={model}, Trial={trial}")
    print(f"Sources: {', '.join(sources)}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "run_ensemble_experiment.py",
        "--dataset", DATASET,
        "--target", target,
        "--sources", *sources,
        "--model", model_full,
        "--trial", str(trial),
        "--seed", str(args.seed),
    ]
    
    if args.deterministic:
        cmd.append("--deterministic")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Experiment failed")
        return None
    
    # Load results
    model_prefix = standardize_model_name(model)
    experiment_name = f"{model_prefix}_{target}_ensemble_trial{trial}"
    results_file = Path("results") / DATASET / experiment_name / "metrics.json"
    
    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def calculate_statistics(trial_results: List[Dict]) -> Dict:
    """Calculate statistics across trials."""
    if not trial_results:
        return {}
    
    stats = {}
    
    # Performance metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        values = [r['test_metrics'][metric] for r in trial_results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    # Time statistics
    times = [r['total_time_minutes'] for r in trial_results]
    stats['time_minutes'] = {
        'mean': np.mean(times),
        'std': np.std(times),
    }
    
    # GPU memory statistics
    if 'gpu_memory' in trial_results[0]:
        memory_values = [r['gpu_memory']['peak_gpu_memory_gb'] for r in trial_results 
                        if r.get('gpu_memory', {}).get('is_cuda', False)]
        if memory_values:
            stats['gpu_memory'] = {
                'mean_gb': np.mean(memory_values),
                'std_gb': np.std(memory_values),
                'min_gb': np.min(memory_values),
                'max_gb': np.max(memory_values),
                'is_cuda': True
            }
        else:
            stats['gpu_memory'] = {
                'mean_gb': 0,
                'std_gb': 0,
                'min_gb': 0,
                'max_gb': 0,
                'is_cuda': False
            }
    
    # FLOPs statistics
    if 'flops' in trial_results[0] and 'error' not in trial_results[0]['flops']:
        flops_values = [r['flops']['total_gflops'] for r in trial_results 
                       if 'flops' in r and 'error' not in r['flops']]
        if flops_values:
            stats['flops'] = {
                'mean_gflops': np.mean(flops_values),
                'std_gflops': np.std(flops_values),
                'min_gflops': np.min(flops_values),
                'max_gflops': np.max(flops_values),
                'available': True
            }
            # Also get breakdown from first result
            if 'val_inference_gflops' in trial_results[0]['flops']:
                stats['flops']['val_inference_mean'] = np.mean([r['flops']['val_inference_gflops'] 
                                                                for r in trial_results if 'flops' in r])
                stats['flops']['test_inference_mean'] = np.mean([r['flops']['test_inference_gflops'] 
                                                                 for r in trial_results if 'flops' in r])
        else:
            stats['flops'] = {'available': False}
    else:
        stats['flops'] = {'available': False}
    
    return stats


def write_summary_report(all_results: Dict, summary_file: Path, args):
    """Write comprehensive summary report."""
    with open(summary_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("ENSEMBLE - CLEAR-100 COMPREHENSIVE RESULTS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Dataset:           CLEAR-100 (30-class subset)\n")
        f.write(f"Model:             {args.model}\n")
        f.write(f"Method:            ENSEMBLE (Soft Voting)\n")
        f.write(f"Trials per Target: {args.num_trials}\n")
        f.write(f"Total Targets:     {len(ALL_YEARS)}\n")
        f.write("\n" + "=" * 100 + "\n\n")
        
        f.write("RESULTS BY TARGET\n")
        f.write("=" * 100 + "\n\n")
        
        for target in ALL_YEARS:
            if target not in all_results:
                continue
            
            trial_results = all_results[target]
            stats = calculate_statistics(trial_results)
            sources = [y for y in ALL_YEARS if y != target]
            
            f.write(f"TARGET: {target.upper()}\n")
            f.write(f"Sources: {', '.join(sources)}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Trials: {len(trial_results)}/{args.num_trials}\n\n")
            
            if stats:
                f.write("Test Performance (Mean +/- Std):\n")
                f.write(f"  Accuracy:  {stats['accuracy']['mean']*100:6.2f}% +/- {stats['accuracy']['std']*100:5.2f}%\n")
                f.write(f"  Precision: {stats['precision']['mean']:6.4f} +/- {stats['precision']['std']:5.4f}\n")
                f.write(f"  Recall:    {stats['recall']['mean']:6.4f} +/- {stats['recall']['std']:5.4f}\n")
                f.write(f"  F1-Score:  {stats['f1_score']['mean']:6.4f} +/- {stats['f1_score']['std']:5.4f}\n\n")
                
                if stats.get('flops', {}).get('available', False):
                    f.write("FLOPs (Mean +/- Std):\n")
                    f.write(f"  Total GFLOPs:          {stats['flops']['mean_gflops']:.2f} +/- {stats['flops']['std_gflops']:.2f}\n")
                    if 'val_inference_mean' in stats['flops']:
                        f.write(f"  Validation Inference:  {stats['flops']['val_inference_mean']:.2f} GFLOPs\n")
                        f.write(f"  Test Inference:        {stats['flops']['test_inference_mean']:.2f} GFLOPs\n")
                    f.write("\n")
                else:
                    f.write("FLOPs: Not available\n\n")
                
                if 'gpu_memory' in stats and stats['gpu_memory']['is_cuda']:
                    f.write("GPU Memory Usage (Mean +/- Std):\n")
                    f.write(f"  Peak Allocated:        {stats['gpu_memory']['mean_gb']:.4f} +/- {stats['gpu_memory']['std_gb']:.4f} GB\n")
                    f.write(f"  Min Peak:              {stats['gpu_memory']['min_gb']:.4f} GB\n")
                    f.write(f"  Max Peak:              {stats['gpu_memory']['max_gb']:.4f} GB\n\n")
                else:
                    f.write("GPU Memory: Not available (CPU mode)\n\n")
                
                f.write(f"Average Time: {stats['time_minutes']['mean']:.2f} +/- {stats['time_minutes']['std']:.2f} minutes\n\n")
            
            f.write("\n")
        
        # Overall summary
        f.write("=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        all_accs = []
        all_memory = []
        all_times = []
        all_flops = []
        
        for results in all_results.values():
            all_accs.extend([r['test_metrics']['accuracy'] for r in results])
            all_times.extend([r['total_time_minutes'] for r in results])
            if results and 'gpu_memory' in results[0]:
                all_memory.extend([r['gpu_memory']['peak_allocated_gb'] for r in results 
                                  if r.get('gpu_memory', {}).get('is_cuda', False)])
            if results and 'flops' in results[0] and 'error' not in results[0]['flops']:
                all_flops.extend([r['flops']['total_gflops'] for r in results 
                                 if 'flops' in r and 'error' not in r['flops']])
        
        f.write(f"Total Experiments:  {len(all_accs)}\n\n")
        
        f.write("Performance:\n")
        f.write(f"  Mean Accuracy:      {np.mean(all_accs)*100:.2f}% +/- {np.std(all_accs)*100:.2f}%\n")
        f.write(f"  Min Accuracy:       {np.min(all_accs)*100:.2f}%\n")
        f.write(f"  Max Accuracy:       {np.max(all_accs)*100:.2f}%\n\n")
        
        if all_flops:
            f.write("FLOPs:\n")
            f.write(f"  Mean Total:         {np.mean(all_flops):.2f} GFLOPs\n")
            f.write(f"  Std Total:          {np.std(all_flops):.2f} GFLOPs\n")
            f.write(f"  Min Total:          {np.min(all_flops):.2f} GFLOPs\n")
            f.write(f"  Max Total:          {np.max(all_flops):.2f} GFLOPs\n\n")
        else:
            f.write("FLOPs: Not available\n\n")
        
        if all_memory:
            f.write("GPU Memory:\n")
            f.write(f"  Mean Peak:          {np.mean(all_memory):.4f} GB\n")
            f.write(f"  Std Peak:           {np.std(all_memory):.4f} GB\n")
            f.write(f"  Min Peak:           {np.min(all_memory):.4f} GB\n")
            f.write(f"  Max Peak:           {np.max(all_memory):.4f} GB\n\n")
        
        f.write("Time:\n")
        f.write(f"  Mean Time:          {np.mean(all_times):.2f} minutes\n")
        f.write(f"  Total Time:         {np.sum(all_times):.2f} minutes ({np.sum(all_times)/60:.2f} hours)\n\n")
        
        f.write("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run all ENSEMBLE experiments for CLEAR-100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='efficientnet-b1',
                        help='Model architecture')
    parser.add_argument('--num_trials', type=int, default=3,
                        help='Number of trials per configuration')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic algorithms')
    
    args = parser.parse_args()

    # Generate random seed if not provided
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Using randomly generated seed: {args.seed}")
    
    print(f"\n{'='*100}")
    print(f"ENSEMBLE COMPREHENSIVE EXPERIMENTS - CLEAR-100")
    print(f"{'='*100}")
    print(f"Model:   {args.model}")
    print(f"Targets: {len(ALL_YEARS)}")
    print(f"Trials:  {args.num_trials} per target")
    print(f"Total:   {len(ALL_YEARS) * args.num_trials} experiments")
    print(f"Classes: 30 (selected subset from 100 total)")
    print(f"{'='*100}\n")
    
    all_results = defaultdict(list)
    
    for target in ALL_YEARS:
        sources = [y for y in ALL_YEARS if y != target]
        
        print(f"\n{'#'*100}")
        print(f"# Processing Target: {target}")
        print(f"# Sources: {', '.join(sources)}")
        print(f"{'#'*100}")
        
        for trial in range(1, args.num_trials + 1):
            result = run_single_experiment(
                target=target,
                sources=sources,
                model=args.model,
                trial=trial,
                args=args
            )
            
            if result:
                all_results[target].append(result)
    
    if all_results:
        summary_file = Path(f"ensemble_{args.model}_clear100_summary.txt")
        write_summary_report(all_results, summary_file, args)
        print(f"\nSummary saved to: {summary_file}")
        
        # Print quick summary
        print(f"\n{'='*100}")
        print("QUICK SUMMARY")
        print(f"{'='*100}")
        for target in ALL_YEARS:
            if target in all_results:
                stats = calculate_statistics(all_results[target])
                if stats:
                    flops_str = ""
                    if stats.get('flops', {}).get('available', False):
                        flops_str = f"  FLOPs: {stats['flops']['mean_gflops']:.2f} GFLOPs"
                    print(f"{target:12s}: {stats['accuracy']['mean']*100:6.2f}% +/- {stats['accuracy']['std']*100:5.2f}%{flops_str}")
        print(f"{'='*100}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
