#!/usr/bin/env python
"""
run_all_fusion_clear10.py - Run all FUSION experiments for CLEAR-10

Runs comprehensive FUSION experiments across CLEAR-10 time periods.

Usage:
    python run_all_fusion_clear10.py --trial 3
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


# CLEAR-10 time periods
ALL_YEARS = ["year_1-2", "year_3-4", "year_5-6", "year_7-8", "year_9-10"]
DATASET = "clear10"


def run_single_experiment(target: str, sources: List[str], model: str, trial: int, args) -> Dict:
    """Run a single fusion experiment."""
    model_full = get_huggingface_model_name(model)
    
    print(f"\n{'='*80}")
    print(f"Running: Target={target}, Model={model}, Trial={trial}")
    print(f"Sources: {', '.join(sources)}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "run_fusion_experiment.py",
        "--dataset", DATASET,
        "--target", target,
        "--sources", *sources,
        "--model", model_full,
        "--trial", str(trial),
        "--seed", str(args.seed),
        "--finetune_epochs", str(args.finetune_epochs),
    ]
    
    if args.deterministic:
        cmd.append("--deterministic")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Experiment failed")
        return None
    
    # Load results
    model_prefix = standardize_model_name(model)
    experiment_name = f"{model_prefix}_{target}_fusion_trial{trial}"
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
            if 'merging_gflops' in trial_results[0]['flops']:
                stats['flops']['merging_mean'] = np.mean([r['flops']['merging_gflops'] 
                                                          for r in trial_results if 'flops' in r])
                stats['flops']['training_mean'] = np.mean([r['flops']['training_gflops'] 
                                                           for r in trial_results if 'flops' in r])
                stats['flops']['inference_mean'] = np.mean([r['flops']['inference_gflops'] 
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
        f.write("FUSION - CLEAR-10 COMPREHENSIVE RESULTS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Dataset:           CLEAR-10 (10-class classification)\n")
        f.write(f"Model:             {args.model}\n")
        f.write(f"Method:            FUSION\n")
        f.write(f"Trials per Target: {args.trial}\n")
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
            f.write(f"Trials: {len(trial_results)}/{args.trial}\n\n")
            
            if stats:
                f.write("Test Performance (Mean +/- Std):\n")
                f.write(f"  Accuracy:  {stats['accuracy']['mean']*100:6.2f}% +/- {stats['accuracy']['std']*100:5.2f}%\n")
                f.write(f"  Precision: {stats['precision']['mean']:6.4f} +/- {stats['precision']['std']:5.4f}\n")
                f.write(f"  Recall:    {stats['recall']['mean']:6.4f} +/- {stats['recall']['std']:5.4f}\n")
                f.write(f"  F1-Score:  {stats['f1_score']['mean']:6.4f} +/- {stats['f1_score']['std']:5.4f}\n\n")
                
                if stats.get('flops', {}).get('available', False):
                    f.write("FLOPs (Mean +/- Std):\n")
                    f.write(f"  Total GFLOPs:          {stats['flops']['mean_gflops']:.2f} +/- {stats['flops']['std_gflops']:.2f}\n")
                    if 'merging_mean' in stats['flops']:
                        f.write(f"  Merging GFLOPs:        {stats['flops']['merging_mean']:.4f}\n")
                        f.write(f"  Training GFLOPs:       {stats['flops']['training_mean']:.2f}\n")
                        f.write(f"  Inference GFLOPs:      {stats['flops']['inference_mean']:.2f}\n")
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
        description="Run all FUSION experiments for CLEAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='mobilevit-xxs',
                        help='Model name (any format accepted by model_utils)')
    parser.add_argument('--trial', type=int, default=3)
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    
    args = parser.parse_args()

    # Generate random seed if not provided
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"Using randomly generated seed: {args.seed}")
    
    print(f"\n{'='*100}")
    print(f"FUSION COMPREHENSIVE EXPERIMENTS - CLEAR-10")
    print(f"{'='*100}")
    print(f"Model:   {args.model}")
    print(f"Targets: {len(ALL_YEARS)}")
    print(f"Trials:  {args.trial} per target")
    print(f"Total:   {len(ALL_YEARS) * args.trial} experiments")
    print(f"{'='*100}\n")
    
    all_results = defaultdict(list)
    
    for target in ALL_YEARS:
        sources = [y for y in ALL_YEARS if y != target]
        
        for trial in range(1, args.trial + 1):
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
        summary_file = Path(f"fusion_{args.model}_clear10_summary.txt")
        write_summary_report(all_results, summary_file, args)
        print(f"\nSummary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
