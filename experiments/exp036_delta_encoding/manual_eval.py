#!/usr/bin/env python3
"""Manually evaluate the trained delta-encoded model"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyphysics import run_rollout

# Create a controller that uses the learned model
class DeltaEncodedController:
    def __init__(self, model_path):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create simple network to match training
        from train_ppo import ActorCritic, STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, CRITIC_LAYERS
        self.network = ActorCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_LAYERS, CRITIC_LAYERS)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, cost {checkpoint['cost']:.2f}")
        
        # PID parameters
        self.pid_p, self.pid_i, self.pid_d = 0.195, 0.100, -0.053
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_residual = 0.0
        self.step_count = 0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.pid_p * error + self.pid_i * self.error_integral + self.pid_d * error_diff
        pid_action = np.clip(pid_action, -2.0, 2.0)
        
        # Residual (if past warmup and have future_plan)
        if self.step_count >= 50 or future_plan is None or len(future_plan.lataccel) == 0:
            scaled_residual = 0.0
        else:
            # Build delta-encoded state
            delta_lataccel = [(lat - target_lataccel) / 3.0 for lat in future_plan.lataccel[:50]]
            delta_roll = [(roll - state.roll_lataccel) / 3.0 for roll in future_plan.roll_lataccel[:50]]
            delta_v = [(v - state.v_ego) / 40.0 for v in future_plan.v_ego[:50]]
            delta_a = [(a - state.a_ego) / 3.0 for a in future_plan.a_ego[:50]]
            
            # Pad to 50 if needed
            while len(delta_lataccel) < 50:
                delta_lataccel.append(delta_lataccel[-1] if delta_lataccel else 0.0)
                delta_roll.append(delta_roll[-1] if delta_roll else 0.0)
                delta_v.append(delta_v[-1] if delta_v else 0.0)
                delta_a.append(delta_a[-1] if delta_a else 0.0)
            
            state_vec = np.array(delta_lataccel + delta_roll + delta_v + delta_a, dtype=np.float32)
            
            # Get residual from network
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
                raw_residual, _, _ = self.network(state_tensor)
                raw_residual = raw_residual.item()
            
            # Process residual
            residual = np.tanh(raw_residual) * 0.5
            filtered = 0.05 * residual + 0.95 * self.prev_residual
            self.prev_residual = filtered
            scaled_residual = filtered * 0.1
        
        self.step_count += 1
        combined = pid_action + scaled_residual
        return float(np.clip(combined, -2.0, 2.0))

# Test on 20 routes
model_path = "ppo_best.pth"
data_files = sorted(Path("../../data").glob("*.csv"))[15000:15020]

print("\nEvaluating delta-encoded model on 20 test routes...")
print("=" * 80)

costs = []
for i, file_path in enumerate(data_files):
    controller = DeltaEncodedController(model_path)
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
    model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
    sim = TinyPhysicsSimulator(model, str(file_path), controller=controller, debug=False)
    cost_dict = sim.rollout()
    costs.append(cost_dict['total_cost'])
    print(f"Route {i+1:2d}: {cost_dict['total_cost']:7.2f}")

print("=" * 80)
print(f"\nDelta-encoded model:")
print(f"  Mean: {np.mean(costs):.2f}")
print(f"  Std:  {np.std(costs):.2f}")

print(f"\nPID baseline:")
print(f"  Mean: 101.31")

print(f"\nDifference: {np.mean(costs) - 101.31:+.2f}")
if np.mean(costs) < 90:
    print("✅✅✅ DELTA-ENCODING HELPS! Cost < 90!")
elif np.mean(costs) < 101:
    print("✅ Delta-encoding helps slightly")
else:
    print("❌ Delta-encoding doesn't help")

