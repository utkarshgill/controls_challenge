#!/usr/bin/env python3
"""
Simple Experiment Harness - Track BC/PPO experiments systematically

Inspired by MLflow/W&B but minimal:
- Logs every run with hyperparameters
- Saves results to JSON
- Easy comparison between runs
- No external dependencies
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import os

class ExperimentTracker:
    """
    Simple experiment tracker for BC/PPO runs
    
    Usage:
        tracker = ExperimentTracker("bc_experiments")
        
        with tracker.run("bc_1000_files") as run:
            run.log_params({
                "n_files": 1000,
                "n_epochs": 50,
                "lr": 1e-3,
            })
            
            # ... train model ...
            
            run.log_metrics({
                "bc_train_cost": 114.43,
                "bc_val_cost": 398.91,
                "pid_baseline": 146.30,
            })
            
            run.save_artifact("bc_pid_best.pth")
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create experiment subdirectory
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(exist_ok=True)
        
        # Load or create experiment log
        self.log_file = self.exp_dir / "runs.jsonl"
        
    def run(self, run_name: str):
        """Start a new run"""
        return Run(self, run_name)
    
    def list_runs(self):
        """List all runs in this experiment"""
        if not self.log_file.exists():
            return []
        
        runs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                runs.append(json.loads(line))
        return runs
    
    def best_run(self, metric: str, minimize: bool = True):
        """Get the best run by a metric"""
        runs = self.list_runs()
        if not runs:
            return None
        
        runs_with_metric = [r for r in runs if metric in r.get('metrics', {})]
        if not runs_with_metric:
            return None
        
        if minimize:
            return min(runs_with_metric, key=lambda r: r['metrics'][metric])
        else:
            return max(runs_with_metric, key=lambda r: r['metrics'][metric])
    
    def summary(self):
        """Print summary of all runs"""
        runs = self.list_runs()
        if not runs:
            print(f"No runs in experiment '{self.experiment_name}'")
            return
        
        print(f"\n{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Total runs: {len(runs)}")
        print(f"{'='*80}\n")
        
        # Header
        print(f"{'Run':<20} {'Status':<10} {'Duration':<12} {'Key Metrics'}")
        print(f"{'-'*80}")
        
        for run in runs:
            name = run['run_name'][:19]
            status = run['status']
            duration = f"{run['duration']:.1f}s" if 'duration' in run else "N/A"
            
            metrics_str = ""
            if 'metrics' in run and run['metrics']:
                # Show up to 3 key metrics
                metric_items = list(run['metrics'].items())[:3]
                metrics_str = ", ".join([f"{k}={v:.1f}" for k, v in metric_items])
            
            print(f"{name:<20} {status:<10} {duration:<12} {metrics_str}")
        
        print()


class Run:
    """A single experimental run"""
    
    def __init__(self, tracker: ExperimentTracker, run_name: str):
        self.tracker = tracker
        self.run_name = run_name
        self.start_time = time.time()
        
        # Run data
        self.data = {
            'run_name': run_name,
            'experiment': tracker.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'params': {},
            'metrics': {},
            'artifacts': [],
        }
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = tracker.exp_dir / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Starting run: {run_name}")
        print(f"Run directory: {self.run_dir}")
        print(f"{'='*80}\n")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.data['status'] = 'completed'
        else:
            self.data['status'] = 'failed'
            self.data['error'] = str(exc_val)
        
        self.data['duration'] = time.time() - self.start_time
        
        # Save run data
        with open(self.tracker.log_file, 'a') as f:
            f.write(json.dumps(self.data) + '\n')
        
        # Save detailed run info
        with open(self.run_dir / 'run_info.json', 'w') as f:
            json.dump(self.data, f, indent=2)
        
        status_emoji = "✅" if self.data['status'] == 'completed' else "❌"
        print(f"\n{'='*80}")
        print(f"{status_emoji} Run {self.run_name} {self.data['status']} in {self.data['duration']:.1f}s")
        print(f"{'='*80}\n")
        
        return False  # Don't suppress exceptions
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.data['params'].update(params)
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics (costs, accuracies, etc.)"""
        self.data['metrics'].update(metrics)
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()
    
    def log_metric(self, key: str, value: float):
        """Log a single metric"""
        self.log_metrics({key: value})
    
    def save_artifact(self, file_path: str, copy: bool = False):
        """Register an artifact (model checkpoint, etc.)"""
        artifact_path = Path(file_path)
        
        if copy and artifact_path.exists():
            import shutil
            dest = self.run_dir / artifact_path.name
            shutil.copy(artifact_path, dest)
            self.data['artifacts'].append(str(dest))
            print(f"Saved artifact: {dest}")
        else:
            self.data['artifacts'].append(str(artifact_path))
            print(f"Registered artifact: {artifact_path}")
    
    def log_note(self, note: str):
        """Add a text note to the run"""
        if 'notes' not in self.data:
            self.data['notes'] = []
        self.data['notes'].append(note)
        print(f"Note: {note}")


# Convenience functions
def compare_runs(experiment_name: str, metric: str):
    """Compare all runs by a metric"""
    tracker = ExperimentTracker(experiment_name)
    runs = tracker.list_runs()
    
    if not runs:
        print(f"No runs found for experiment: {experiment_name}")
        return
    
    print(f"\nComparing runs by: {metric}")
    print(f"{'='*80}")
    
    runs_with_metric = [(r['run_name'], r['metrics'].get(metric)) 
                        for r in runs if metric in r.get('metrics', {})]
    
    if not runs_with_metric:
        print(f"No runs have metric: {metric}")
        return
    
    runs_with_metric.sort(key=lambda x: x[1])
    
    for name, value in runs_with_metric:
        print(f"{name:<30} {value:.2f}")
    
    print()


if __name__ == '__main__':
    # Demo usage
    tracker = ExperimentTracker("demo_experiment")
    
    # Run 1
    with tracker.run("test_run_1") as run:
        run.log_params({"lr": 0.001, "batch_size": 256})
        time.sleep(0.5)  # Simulate training
        run.log_metrics({"train_loss": 0.123, "val_loss": 0.456})
    
    # Run 2
    with tracker.run("test_run_2") as run:
        run.log_params({"lr": 0.0001, "batch_size": 512})
        time.sleep(0.5)
        run.log_metrics({"train_loss": 0.089, "val_loss": 0.234})
    
    # Show summary
    tracker.summary()
    
    # Show best
    best = tracker.best_run("val_loss", minimize=True)
    print(f"\nBest run by val_loss: {best['run_name']} ({best['metrics']['val_loss']:.3f})")

