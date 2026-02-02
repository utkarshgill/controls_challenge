"""
Compare MLP vs Conv BC models on robustness.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid
from bc_controller import BCController
from bc_controller_conv import BCControllerConv


def main():
    model_path = "../../models/tinyphysics.onnx"
    data_dir = "../../data"
    mlp_checkpoint = "./outputs/best_model.pt"
    conv_checkpoint = "./outputs/best_model_conv.pt"
    
    print("Loading models...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Test on different files (1000-1200)
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    test_files = all_files[1000:1200]
    
    print(f"Testing on {len(test_files)} files...\n")
    
    # Evaluate all three
    pid_costs = []
    mlp_costs = []
    conv_costs = []
    
    for file_path in tqdm(test_files, desc="Evaluating"):
        # PID
        pid_ctrl = pid.Controller()
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=pid_ctrl, debug=False)
        pid_costs.append(sim.rollout()['total_cost'])
        
        # MLP
        mlp_ctrl = BCController(mlp_checkpoint)
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=mlp_ctrl, debug=False)
        mlp_costs.append(sim.rollout()['total_cost'])
        
        # Conv
        conv_ctrl = BCControllerConv(conv_checkpoint)
        sim = TinyPhysicsSimulator(tinyphysics_model, str(file_path), controller=conv_ctrl, debug=False)
        conv_costs.append(sim.rollout()['total_cost'])
    
    pid_costs = np.array(pid_costs)
    mlp_costs = np.array(mlp_costs)
    conv_costs = np.array(conv_costs)
    
    # Print comparison
    print(f"\n{'='*70}")
    print("ROBUSTNESS COMPARISON: MLP vs Conv")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<20} {'PID':<15} {'MLP':<15} {'Conv':<15}")
    print(f"{'-'*65}")
    print(f"{'Mean':<20} {pid_costs.mean():<15.2f} {mlp_costs.mean():<15.2f} {conv_costs.mean():<15.2f}")
    print(f"{'Std':<20} {pid_costs.std():<15.2f} {mlp_costs.std():<15.2f} {conv_costs.std():<15.2f}")
    print(f"{'Median':<20} {np.median(pid_costs):<15.2f} {np.median(mlp_costs):<15.2f} {np.median(conv_costs):<15.2f}")
    print(f"{'95th percentile':<20} {np.percentile(pid_costs, 95):<15.2f} {np.percentile(mlp_costs, 95):<15.2f} {np.percentile(conv_costs, 95):<15.2f}")
    print(f"{'99th percentile':<20} {np.percentile(pid_costs, 99):<15.2f} {np.percentile(mlp_costs, 99):<15.2f} {np.percentile(conv_costs, 99):<15.2f}")
    print(f"{'Max':<20} {pid_costs.max():<15.2f} {mlp_costs.max():<15.2f} {conv_costs.max():<15.2f}")
    
    # Ratios vs PID
    print(f"\n{'='*70}")
    print("PERFORMANCE vs PID")
    print(f"{'='*70}\n")
    
    mlp_ratio = mlp_costs.mean() / pid_costs.mean()
    conv_ratio = conv_costs.mean() / pid_costs.mean()
    
    print(f"Mean cost ratio:")
    print(f"  MLP:  {mlp_ratio:.2f}x PID")
    print(f"  Conv: {conv_ratio:.2f}x PID")
    
    if conv_ratio < mlp_ratio:
        improvement = (1 - conv_ratio/mlp_ratio) * 100
        print(f"\n✓ Conv is {improvement:.1f}% better than MLP")
    else:
        degradation = (conv_ratio/mlp_ratio - 1) * 100
        print(f"\n✗ Conv is {degradation:.1f}% worse than MLP")
    
    # Failure analysis
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS (cost > 200)")
    print(f"{'='*70}\n")
    
    threshold = 200
    pid_failures = np.sum(pid_costs > threshold)
    mlp_failures = np.sum(mlp_costs > threshold)
    conv_failures = np.sum(conv_costs > threshold)
    
    print(f"{'Model':<15} {'Failures':<15} {'Percentage':<15}")
    print(f"{'-'*45}")
    print(f"{'PID':<15} {pid_failures}/{len(test_files):<15} {100*pid_failures/len(test_files):.1f}%")
    print(f"{'MLP':<15} {mlp_failures}/{len(test_files):<15} {100*mlp_failures/len(test_files):.1f}%")
    print(f"{'Conv':<15} {conv_failures}/{len(test_files):<15} {100*conv_failures/len(test_files):.1f}%")
    
    # Where Conv helps most
    improvements = mlp_costs - conv_costs
    big_improvements = np.sum(improvements > 50)
    big_regressions = np.sum(improvements < -50)
    
    print(f"\n{'='*70}")
    print("WHERE CONV MAKES A DIFFERENCE")
    print(f"{'='*70}\n")
    
    print(f"Files where Conv is >50 better than MLP: {big_improvements}")
    print(f"Files where Conv is >50 worse than MLP: {big_regressions}")
    print(f"Mean improvement per file: {improvements.mean():.2f}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")
    
    if conv_ratio < 1.2 and conv_failures <= pid_failures * 1.2:
        print(f"✓✓ Conv is ROBUST: {conv_ratio:.2f}x PID")
        print(f"   → Excellent baseline for PPO!")
    elif conv_ratio < mlp_ratio * 0.9:
        print(f"✓ Conv is better than MLP: {conv_ratio:.2f}x vs {mlp_ratio:.2f}x PID")
        print(f"   → Use Conv for PPO warm-start")
    elif conv_ratio < mlp_ratio * 1.1:
        print(f"~ Conv similar to MLP: {conv_ratio:.2f}x vs {mlp_ratio:.2f}x PID")
        print(f"   → Either works for PPO")
    else:
        print(f"✗ MLP is better: Use MLP for PPO warm-start")
    
    print(f"\nRecommendation: Proceed with PPO using {'Conv' if conv_ratio <= mlp_ratio else 'MLP'} as baseline")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
