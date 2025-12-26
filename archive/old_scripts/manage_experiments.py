#!/usr/bin/env python3
"""
Experiment Management Tool

Usage:
    python manage_experiments.py list
    python manage_experiments.py compare baseline exp001_bc_with_a_ego
    python manage_experiments.py new exp002_my_idea "Test my hypothesis"
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import shutil

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"

def list_experiments():
    """List all experiments with their status"""
    print("\n" + "="*80)
    print("EXPERIMENTS")
    print("="*80)
    
    experiments = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name != "template"])
    
    for exp_dir in experiments:
        readme_path = exp_dir / "README.md"
        metrics_path = exp_dir / "results" / "metrics.json"
        
        # Parse status from README
        status = "❓ Unknown"
        hypothesis = "N/A"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                for line in f:
                    if "**Status**:" in line:
                        if "✅ Complete" in line:
                            status = "✅ Complete"
                        elif "🏃 Running" in line or "🏃 Ready" in line:
                            status = "🏃 Ready"
                        elif "❌ Failed" in line:
                            status = "❌ Failed"
                    if "## Hypothesis" in line:
                        hypothesis = next(f).strip()
                        break
        
        # Get results if available
        result_str = ""
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                if "mean_cost" in metrics:
                    result_str = f" → {metrics['mean_cost']:.1f}"
        
        print(f"\n{exp_dir.name}")
        print(f"  Status: {status}{result_str}")
        print(f"  Hypothesis: {hypothesis[:60]}...")
    
    print("\n" + "="*80)

def compare_experiments(exp_names):
    """Compare results from multiple experiments"""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    results = {}
    for exp_name in exp_names:
        exp_dir = EXPERIMENTS_DIR / exp_name
        metrics_path = exp_dir / "results" / "metrics.json"
        
        if not metrics_path.exists():
            print(f"\n⚠️  {exp_name}: No results found")
            continue
        
        with open(metrics_path, 'r') as f:
            results[exp_name] = json.load(f)
    
    if not results:
        print("\n❌ No results to compare")
        return
    
    # Print comparison table
    print(f"\n{'Experiment':<30} {'Mean':<10} {'Median':<10} {'Failures':<10}")
    print("-" * 80)
    
    for exp_name, metrics in results.items():
        mean = metrics.get('mean_cost', 'N/A')
        median = metrics.get('median_cost', 'N/A')
        failures = metrics.get('failures', 'N/A')
        
        mean_str = f"{mean:.2f}" if isinstance(mean, (int, float)) else mean
        median_str = f"{median:.2f}" if isinstance(median, (int, float)) else median
        failures_str = str(failures)
        
        print(f"{exp_name:<30} {mean_str:<10} {median_str:<10} {failures_str:<10}")
    
    # Find best
    if all(isinstance(results[e].get('mean_cost'), (int, float)) for e in results):
        best_exp = min(results.keys(), key=lambda e: results[e]['mean_cost'])
        best_cost = results[best_exp]['mean_cost']
        print(f"\n🏆 Best: {best_exp} ({best_cost:.2f})")
    
    print("\n" + "="*80)

def create_experiment(exp_name, description):
    """Create a new experiment from template"""
    exp_dir = EXPERIMENTS_DIR / exp_name
    
    if exp_dir.exists():
        print(f"❌ Experiment {exp_name} already exists")
        return
    
    # Copy template
    template_dir = EXPERIMENTS_DIR / "template"
    shutil.copytree(template_dir, exp_dir)
    
    # Update README with experiment name and description
    readme_path = exp_dir / "README.md"
    with open(readme_path, 'r') as f:
        content = f.read()
    
    content = content.replace("Experiment XXX: [Name]", f"Experiment: {exp_name}")
    content = content.replace("What are we testing?", description)
    content = content.replace("YYYY-MM-DD", datetime.now().strftime("%Y-%m-%d"))
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Created experiment: {exp_name}")
    print(f"   Location: {exp_dir}")
    print(f"\nNext steps:")
    print(f"  1. Edit {exp_dir}/README.md")
    print(f"  2. Edit {exp_dir}/config.yaml")
    print(f"  3. Implement {exp_dir}/run.py")
    print(f"  4. Run: cd {exp_dir} && python run.py")

def show_status():
    """Show overall project status"""
    print("\n" + "="*80)
    print("PROJECT STATUS")
    print("="*80)
    
    experiments = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and d.name != "template"]
    
    completed = 0
    running = 0
    failed = 0
    
    best_result = float('inf')
    best_exp = None
    
    for exp_dir in experiments:
        readme_path = exp_dir / "README.md"
        metrics_path = exp_dir / "results" / "metrics.json"
        
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
                if "✅ Complete" in content:
                    completed += 1
                elif "🏃" in content:
                    running += 1
                elif "❌ Failed" in content:
                    failed += 1
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                if "mean_cost" in metrics:
                    cost = metrics['mean_cost']
                    if cost < best_result:
                        best_result = cost
                        best_exp = exp_dir.name
    
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"  ✅ Completed: {completed}")
    print(f"  🏃 Running/Ready: {running}")
    print(f"  ❌ Failed: {failed}")
    
    if best_exp:
        print(f"\n🏆 Best result: {best_exp} ({best_result:.2f})")
        print(f"   Target: < 45.0")
        print(f"   Gap: {best_result - 45.0:.2f} points")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Manage experiments")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    subparsers.add_parser('list', help='List all experiments')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiments', nargs='+', help='Experiment names to compare')
    
    # New command
    new_parser = subparsers.add_parser('new', help='Create new experiment')
    new_parser.add_argument('name', help='Experiment name (e.g., exp002_my_idea)')
    new_parser.add_argument('description', help='Brief description of hypothesis')
    
    # Status command
    subparsers.add_parser('status', help='Show project status')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_experiments()
    elif args.command == 'compare':
        compare_experiments(args.experiments)
    elif args.command == 'new':
        create_experiment(args.name, args.description)
    elif args.command == 'status':
        show_status()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

