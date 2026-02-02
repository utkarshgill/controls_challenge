"""Evaluate exp010 on test set"""
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from train import ShallowActorCritic, build_state

model_path = '/Users/engelbart/Desktop/stuff/controls_challenge/models/tinyphysics.onnx'
data_folder = '/Users/engelbart/Desktop/stuff/controls_challenge/data'

actor_critic = ShallowActorCritic(state_dim=5, action_dim=1, hidden_size=16)
actor_critic.load_state_dict(torch.load('results/ppo_best.pth', weights_only=True))
actor_critic.eval()

files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])[:100]
costs = []

for file in files:
    file_path = os.path.join(data_folder, file)
    model = TinyPhysicsModel(model_path, debug=False)
    sim = TinyPhysicsSimulator(model, file_path, controller=None, debug=False)
    
    prev_error = 0.0
    error_integral = 0.0
    
    while sim.step_idx < len(sim.data) - 50:
        state, target, future_plan = sim.get_state_target_futureplan(sim.step_idx)
        current_lataccel = sim.current_lataccel
        
        obs = build_state(target, current_lataccel, state, prev_error, error_integral, future_plan)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, _ = actor_critic(obs_tensor)
            action = torch.tanh(action_mean).cpu().numpy()[0][0]
        
        sim.action_history.append(action)
        sim.state_history.append(state)
        sim.target_lataccel_history.append(target)
        sim.sim_step(sim.step_idx)
        sim.step_idx += 1
        
        error = target - current_lataccel
        error_integral += error
        error_integral = np.clip(error_integral, -14, 14)
        prev_error = error
    
    cost_dict = sim.compute_cost()
    costs.append(cost_dict['total_cost'])

print(f"exp010 (Shallow, 5 feat + future): {np.mean(costs):.1f}")
