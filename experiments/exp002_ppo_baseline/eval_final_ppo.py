#!/usr/bin/env python3
"""Evaluate the best PPO model"""
import numpy as np
import glob
from tqdm import tqdm
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers.ppo_parallel import Controller as PPOController
from controllers.pid import Controller as PIDController

def evaluate(controller, files, model_path):
    costs = []
    for f in tqdm(files, desc=f"Evaluating"):
        model = TinyPhysicsModel(model_path, debug=False)
        sim = TinyPhysicsSimulator(model, f, controller=controller, debug=False)
        sim.rollout()
        costs.append(sim.compute_cost()['total_cost'])
    return np.array(costs)

print("\n" + "="*60)
print("FINAL PPO EVALUATION")
print("="*60)

model_path = "./models/tinyphysics.onnx"
all_files = sorted(glob.glob("./data/*.csv"))
test_files = all_files[:100]  # First 100 files

print("\nEvaluating PID baseline...")
pid = PIDController()
pid_costs = evaluate(pid, test_files, model_path)

print("\nEvaluating PPO (best)...")
ppo = PPOController()
ppo_costs = evaluate(ppo, test_files, model_path)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Controller':<15} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
print("-" * 60)
print(f"{'PID':<15} {np.mean(pid_costs):<10.2f} {np.median(pid_costs):<10.2f} {np.min(pid_costs):<10.2f} {np.max(pid_costs):<10.2f}")
print(f"{'PPO':<15} {np.mean(ppo_costs):<10.2f} {np.median(ppo_costs):<10.2f} {np.min(ppo_costs):<10.2f} {np.max(ppo_costs):<10.2f}")

print("\n" + "="*60)
print(f"Target: < 45.0")
print(f"PPO:    {np.mean(ppo_costs):.2f} {'✅ SUCCESS!' if np.mean(ppo_costs) < 45 else '❌ FAILED'}")
print(f"Improvement over PID: {((np.mean(pid_costs) - np.mean(ppo_costs)) / np.mean(pid_costs) * 100):.1f}%")
print("="*60)

np.savez('final_ppo_eval.npz', pid=pid_costs, ppo=ppo_costs)
print("\n✅ Saved to final_ppo_eval.npz")

