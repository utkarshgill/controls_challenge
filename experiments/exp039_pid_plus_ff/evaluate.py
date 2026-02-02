"""
Evaluate trained PID+FF model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train import ActorCritic, rollout_single_episode, train_files, eval_files, test_files

# Load model
model_path = Path(__file__).parent / 'best_model.pth'
if not model_path.exists():
    print(f"Model not found: {model_path}")
    print("Train first: python train.py")
    sys.exit(1)

actor_critic = ActorCritic()
actor_critic.load_state_dict(torch.load(model_path, map_location='cpu'))
actor_critic.eval()

print("Evaluating PID+FF model...")
print("="*60)

# Evaluate on splits
for split_name, files in [('Train', train_files[:100]), 
                           ('Eval', eval_files[:100]), 
                           ('Test', test_files[:100])]:
    costs = []
    for data_file in files:
        cost_dict, _ = rollout_single_episode(actor_critic, data_file, deterministic=True)
        costs.append(cost_dict['total_cost'])
    
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    print(f"{split_name:5s}: {mean_cost:6.2f} ± {std_cost:5.2f}  (PID baseline: ~75)")

print("="*60)

# Compare to PID
print("\nComparing to PID baseline...")
from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'tinyphysics.onnx'
tinyphysics_model = TinyPhysicsModel(str(MODEL_PATH), debug=False)

pid_costs = []
for data_file in test_files[:100]:
    controller = PIDController()
    sim = TinyPhysicsSimulator(tinyphysics_model, str(data_file), controller=controller, debug=False)
    cost_dict = sim.rollout()
    pid_costs.append(cost_dict['total_cost'])

pid_mean = np.mean(pid_costs)
pid_std = np.std(pid_costs)

print(f"PID:   {pid_mean:6.2f} ± {pid_std:5.2f}")
print(f"Ours:  {np.mean([cost_dict['total_cost'] for cost_dict in [rollout_single_episode(actor_critic, f, True)[0] for f in test_files[:100]]]):6.2f}")
print(f"Improvement: {pid_mean - np.mean([rollout_single_episode(actor_critic, f, True)[0]['total_cost'] for f in test_files[:100]]):6.2f} points")

