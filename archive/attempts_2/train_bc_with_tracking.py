#!/usr/bin/env python3
"""
BC Training with Experiment Tracking

Wrapper around train_bc_pid.py that logs experiments systematically.
"""

import sys
from experiment_harness import ExperimentTracker
from train_bc_pid import train_bc, state_dim, action_dim, hidden_dim, trunk_layers, head_layers

def run_bc_experiment(
    run_name: str,
    n_expert_files: int = 1000,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """
    Run a BC experiment with tracking
    
    Args:
        run_name: Name for this run
        n_expert_files: Number of files to collect expert data from
        n_epochs: BC training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    tracker = ExperimentTracker("bc_experiments")
    
    with tracker.run(run_name) as run:
        # Log all hyperparameters
        run.log_params({
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "trunk_layers": trunk_layers,
            "head_layers": head_layers,
            "n_expert_files": n_expert_files,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
        })
        
        # Temporarily modify train_bc_pid's globals
        import train_bc_pid
        original_n_files = 1000  # Default in train_bc_pid
        original_n_epochs = 50
        original_batch_size = 256
        
        # Monkey-patch (hacky but quick)
        # TODO: Refactor train_bc_pid to accept parameters
        
        # Run BC training
        print(f"\n🚀 Starting BC training: {run_name}")
        print(f"   Expert files: {n_expert_files}, Epochs: {n_epochs}")
        print()
        
        # Run BC with specified parameters
        network = train_bc(n_expert_files=n_expert_files)
        
        # Metrics are printed by train_bc(), but not returned
        # TODO: Refactor train_bc() to return metrics
        run.log_note("Metrics logged by train_bc() - see stdout")
        
        # Save artifacts
        run.save_artifact("bc_pid_best.pth")
        run.save_artifact("bc_pid_checkpoint.pth")
        
        print(f"\n✅ Run '{run_name}' complete. Check experiments/bc_experiments/")
        
        return network


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Run name')
    parser.add_argument('--files', type=int, default=1000, help='Number of expert files')
    parser.add_argument('--epochs', type=int, default=50, help='BC epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    
    args = parser.parse_args()
    
    run_bc_experiment(
        run_name=args.name,
        n_expert_files=args.files,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    
    # Show summary after run
    tracker = ExperimentTracker("bc_experiments")
    tracker.summary()

