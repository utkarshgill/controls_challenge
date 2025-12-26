#!/usr/bin/env python3
"""
Evaluate Experiment 001: BC with a_ego
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller as PIDController

# Import training code for state builder and network
from train import BCNetwork, build_state, OBS_SCALE

class BCWithAEgoController:
    """Controller using BC model with a_ego"""
    def __init__(self, checkpoint_path="results/checkpoints/bc_with_a_ego.pt"):
        self.model = BCNetwork(state_dim=57)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()
        
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral = np.clip(self.error_integral + error, -14, 14)
        
        state_vec = build_state(target_lataccel, current_lataccel, state, future_plan, 
                                self.prev_error, self.error_integral)
        
        # MULTIPLY by OBS_SCALE (not divide!)
        state_normalized = torch.from_numpy(state_vec * OBS_SCALE).float().unsqueeze(0)
        
        with torch.no_grad():
            action = self.model(state_normalized).squeeze().item()
        
        self.prev_error = error
        return action

def evaluate_controller(controller, files):
    """Evaluate controller on files"""
    costs = []
    model_path = "../../models/tinyphysics.onnx"
    
    for file_path in tqdm(files, desc="Evaluating"):
        model = TinyPhysicsModel(model_path, debug=False)
        sim = TinyPhysicsSimulator(model, file_path, controller=controller, debug=False)
        sim.rollout()
        cost = sim.compute_cost()
        costs.append(cost['total_cost'])
    
    return np.array(costs)

def main():
    print("\n" + "="*60)
    print("EXPERIMENT 001: Evaluation")
    print("="*60)
    
    # Test files (same 100 as baseline)
    all_files = sorted(glob.glob("../../data/*.csv"))
    test_files = all_files[:100]
    
    # Baseline (for comparison)
    print("\n[1/2] Running PID baseline...")
    pid = PIDController()
    pid_costs = evaluate_controller(pid, test_files)
    
    # Experiment 001
    print("\n[2/2] Running BC with a_ego...")
    bc_with_a_ego = BCWithAEgoController()
    exp_costs = evaluate_controller(bc_with_a_ego, test_files)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Controller':<20} {'Mean':<10} {'Median':<10} {'Std':<10}")
    print("-" * 60)
    print(f"{'PID (baseline)':<20} {np.mean(pid_costs):<10.2f} {np.median(pid_costs):<10.2f} {np.std(pid_costs):<10.2f}")
    print(f"{'BC + a_ego':<20} {np.mean(exp_costs):<10.2f} {np.median(exp_costs):<10.2f} {np.std(exp_costs):<10.2f}")
    
    # Specific file analysis
    print("\n" + "="*60)
    print("FILE 00069 (Hard case)")
    print("="*60)
    file_69_idx = 69
    print(f"PID:         {pid_costs[file_69_idx]:.2f}")
    print(f"BC + a_ego:  {exp_costs[file_69_idx]:.2f}")
    improvement = pid_costs[file_69_idx] - exp_costs[file_69_idx]
    print(f"Improvement: {improvement:+.2f}")
    
    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    mean_cost = np.mean(exp_costs)
    file_69_cost = exp_costs[file_69_idx]
    
    success = []
    if mean_cost < 90:
        success.append("✅ Mean < 90")
    else:
        success.append(f"❌ Mean >= 90 (got {mean_cost:.2f})")
    
    if file_69_cost < 800:
        success.append("✅ File 00069 < 800")
    else:
        success.append(f"❌ File 00069 >= 800 (got {file_69_cost:.2f})")
    
    for s in success:
        print(f"  {s}")
    
    # Save results
    np.savez('results/eval_results.npz', 
             pid=pid_costs, 
             bc_with_a_ego=exp_costs)
    print("\n✅ Results saved to results/eval_results.npz")
    
    # Conclusion
    print("\n" + "="*60)
    if all("✅" in s for s in success):
        print("🎉 EXPERIMENT SUCCESS!")
        print("   a_ego hypothesis CONFIRMED")
        print("   Next: Try PPO with a_ego")
    elif any("✅" in s for s in success):
        print("⚠️  PARTIAL SUCCESS")
        print("   a_ego helps but not enough")
        print("   Next: Try adding friction margin")
    else:
        print("❌ EXPERIMENT FAILED")
        print("   a_ego hypothesis rejected")
        print("   Next: Debug or try different approach")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

