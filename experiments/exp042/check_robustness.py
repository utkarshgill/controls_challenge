"""
Check robustness: Look at worst-case scenarios for BC vs PID.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid
from bc_controller import BCController


def main():
    # Load model
    model_path = "../../models/tinyphysics.onnx"
    data_dir = "../../data"
    bc_checkpoint = "./outputs/best_model.pt"
    
    print("Loading models...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Get test files (different subset than training)
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    test_files = all_files[1000:1200]  # Different 200 files
    
    print(f"Testing on {len(test_files)} files...\n")
    
    # Evaluate both controllers
    pid_costs = []
    bc_costs = []
    
    for file_path in tqdm(test_files, desc="Evaluating"):
        # PID
        pid_ctrl = pid.Controller()
        sim_pid = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=pid_ctrl, debug=False)
        cost_pid = sim_pid.rollout()
        pid_costs.append(cost_pid['total_cost'])
        
        # BC
        bc_ctrl = BCController(bc_checkpoint)
        sim_bc = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=bc_ctrl, debug=False)
        cost_bc = sim_bc.rollout()
        bc_costs.append(cost_bc['total_cost'])
    
    pid_costs = np.array(pid_costs)
    bc_costs = np.array(bc_costs)
    
    # Statistics
    print(f"\n{'='*70}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<20} {'PID':<20} {'BC':<20}")
    print(f"{'-'*60}")
    print(f"{'Mean':<20} {pid_costs.mean():<20.2f} {bc_costs.mean():<20.2f}")
    print(f"{'Std':<20} {pid_costs.std():<20.2f} {bc_costs.std():<20.2f}")
    print(f"{'Min':<20} {pid_costs.min():<20.2f} {bc_costs.min():<20.2f}")
    print(f"{'Max':<20} {pid_costs.max():<20.2f} {bc_costs.max():<20.2f}")
    print(f"{'Median':<20} {np.median(pid_costs):<20.2f} {np.median(bc_costs):<20.2f}")
    print(f"{'95th percentile':<20} {np.percentile(pid_costs, 95):<20.2f} {np.percentile(bc_costs, 95):<20.2f}")
    print(f"{'99th percentile':<20} {np.percentile(pid_costs, 99):<20.2f} {np.percentile(bc_costs, 99):<20.2f}")
    
    # Failure analysis
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS")
    print(f"{'='*70}\n")
    
    threshold = 200
    pid_failures = np.sum(pid_costs > threshold)
    bc_failures = np.sum(bc_costs > threshold)
    
    print(f"Files with cost > {threshold}:")
    print(f"  PID: {pid_failures}/{len(test_files)} ({100*pid_failures/len(test_files):.1f}%)")
    print(f"  BC:  {bc_failures}/{len(test_files)} ({100*bc_failures/len(test_files):.1f}%)")
    
    # Where BC is worse than PID
    worse_mask = bc_costs > pid_costs * 1.5  # BC is 50% worse
    num_worse = np.sum(worse_mask)
    print(f"\nFiles where BC is 50% worse than PID:")
    print(f"  Count: {num_worse}/{len(test_files)} ({100*num_worse/len(test_files):.1f}%)")
    
    if num_worse > 0:
        print(f"  Mean BC cost on these: {bc_costs[worse_mask].mean():.2f}")
        print(f"  Mean PID cost on these: {pid_costs[worse_mask].mean():.2f}")
    
    # Where BC is better than PID
    better_mask = bc_costs < pid_costs * 0.8  # BC is 20% better
    num_better = np.sum(better_mask)
    print(f"\nFiles where BC is 20% better than PID:")
    print(f"  Count: {num_better}/{len(test_files)} ({100*num_better/len(test_files):.1f}%)")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")
    
    ratio = bc_costs.mean() / pid_costs.mean()
    if ratio < 1.1 and bc_failures <= pid_failures * 1.2:
        print(f"✓ BC is ROBUST: {ratio:.2f}x PID with similar failure rate")
        print(f"  → Good baseline for PPO fine-tuning")
    elif ratio < 1.5:
        print(f"⚠ BC is ACCEPTABLE: {ratio:.2f}x PID")
        print(f"  → Can proceed with PPO, expect improvements")
    else:
        print(f"✗ BC needs work: {ratio:.2f}x PID")
        print(f"  → Consider more training or architecture changes")


if __name__ == '__main__':
    main()
