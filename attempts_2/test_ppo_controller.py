#!/usr/bin/env python3
"""
Test the PPO controller on validation set.
"""
import numpy as np
import torch
import glob
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, STEER_RANGE

# State building (must match training)
OBS_SCALE = np.array(
    [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] + [1000.0] * 50,
    dtype=np.float32
)

def build_state(target_lataccel, current_lataccel, state, future_plan, prev_error, error_integral):
    """56D: PID terms (3) + current state (3) + 50 future curvatures"""
    eps = 1e-6
    error = target_lataccel - current_lataccel
    error_diff = error - prev_error
    
    curv_now = (target_lataccel - state.roll_lataccel) / (state.v_ego ** 2 + eps)
    
    future_curvs = []
    for t in range(min(50, len(future_plan.lataccel))):
        lat = future_plan.lataccel[t]
        roll = future_plan.roll_lataccel[t]
        v = future_plan.v_ego[t]
        curv = (lat - roll) / (v ** 2 + eps)
        future_curvs.append(curv)
    
    while len(future_curvs) < 50:
        future_curvs.append(0.0)
    
    state_vec = [error, error_diff, error_integral, current_lataccel, state.v_ego, curv_now] + future_curvs
    return np.array(state_vec, dtype=np.float32)

class PPOController:
    def __init__(self, model_path):
        from train_ppo_parallel import ActorCritic
        self.actor_critic = ActorCritic(56, 1, 128, 1, 3)
        self.actor_critic.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.actor_critic.eval()
        
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def reset(self):
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        obs = build_state(target_lataccel, current_lataccel, state, future_plan,
                         self.prev_error, self.error_integral)
        
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs * OBS_SCALE, dtype=torch.float32).unsqueeze(0)
            action_mean, _, _ = self.actor_critic(obs_tensor)
            raw_action = action_mean  # Deterministic
            action = torch.tanh(raw_action) * STEER_RANGE[1]
        
        # Update state
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.prev_error = error
        
        return action.item()

def evaluate_controller(controller_path, data_files, n_files=100):
    """Evaluate controller on n_files"""
    costs = []
    print("\nLoading model...")
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
    print("Loading controller...")
    controller = PPOController(controller_path)
    print(f"Testing on {n_files} files...\n")
    
    from tqdm import tqdm
    for i, data_file in enumerate(tqdm(data_files[:n_files], desc="Evaluating")):
        controller.reset()
        sim = TinyPhysicsSimulator(model, data_file, controller=controller, debug=False)
        sim.rollout()
        cost = sim.compute_cost()
        costs.append(cost['total_cost'])
        
        # Safety check
        if i < 5:
            print(f"  File {i}: {cost['total_cost']:.2f}")
    
    return costs

if __name__ == '__main__':
    import os
    import sys
    
    print("\n" + "="*60)
    print("Testing PPO Controller")
    print("="*60)
    
    # Check if weights exist
    if not os.path.exists('ppo_parallel_best.pth'):
        print("❌ ppo_parallel_best.pth not found!")
        sys.exit(1)
    
    # Load validation files
    all_files = sorted(glob.glob("./data/*.csv"))
    if not all_files:
        print("❌ No data files found!")
        sys.exit(1)
    
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.9)
    val_files = all_files[split_idx:]
    
    print(f"\nTotal files: {len(all_files)}")
    print(f"Validation files: {len(val_files)}")
    
    costs = evaluate_controller('ppo_parallel_best.pth', val_files, n_files=100)
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Mean cost: {np.mean(costs):.2f}")
    print(f"Median cost: {np.median(costs):.2f}")
    print(f"Min cost: {np.min(costs):.2f}")
    print(f"Max cost: {np.max(costs):.2f}")
    print(f"Std: {np.std(costs):.2f}")
    
    # Show distribution
    print(f"\nDistribution:")
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    for i in range(len(bins)-1):
        count = sum(bins[i] <= c < bins[i+1] for c in costs)
        print(f"  [{bins[i]:4.0f}, {bins[i+1]:4.0f}): {count:3d} files")

