#!/usr/bin/env python3
"""
Compare Experiments - Analyze all BC/PPO runs

Usage:
    python compare_experiments.py bc              # Show all BC runs
    python compare_experiments.py bc --metric bc_val_cost  # Sort by specific metric
    python compare_experiments.py bc --best       # Show only best run
"""

import argparse
from experiment_harness import ExperimentTracker, compare_runs

def main():
    parser = argparse.ArgumentParser(description='Compare experimental runs')
    parser.add_argument('experiment', choices=['bc', 'ppo', 'bc_ppo'], 
                       help='Which experiment to analyze')
    parser.add_argument('--metric', type=str, help='Metric to compare (e.g., bc_val_cost)')
    parser.add_argument('--best', action='store_true', help='Show only best run')
    
    args = parser.parse_args()
    
    experiment_map = {
        'bc': 'bc_experiments',
        'ppo': 'ppo_experiments', 
        'bc_ppo': 'bc_ppo_experiments',
    }
    
    exp_name = experiment_map[args.experiment]
    tracker = ExperimentTracker(exp_name)
    
    if args.best and args.metric:
        best = tracker.best_run(args.metric, minimize=True)
        if best:
            print(f"\nBest run by {args.metric}:")
            print(f"  Name: {best['run_name']}")
            print(f"  {args.metric}: {best['metrics'][args.metric]:.2f}")
            print(f"  Params: {best['params']}")
        else:
            print(f"No runs with metric: {args.metric}")
    
    elif args.metric:
        compare_runs(exp_name, args.metric)
    
    else:
        tracker.summary()

if __name__ == '__main__':
    main()

