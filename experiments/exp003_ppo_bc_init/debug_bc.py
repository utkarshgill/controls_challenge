#!/usr/bin/env python3
"""Debug BC vs PID on single file"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers.pid import Controller as PIDController
from controllers.bc import Controller as BCController
import matplotlib.pyplot as plt

def main():
    model_path = "../../models/tinyphysics.onnx"
    file_path = "../../data/00000.csv"
    
    print("="*60)
    print("DEBUG: BC vs PID on single file")
    print("="*60)
    
    # Run PID
    print("\n[1/2] Running PID...")
    model = TinyPhysicsModel(model_path, debug=False)
    pid = PIDController()
    sim_pid = TinyPhysicsSimulator(model, file_path, controller=pid, debug=False)
    sim_pid.rollout()
    pid_cost = sim_pid.compute_cost()
    
    print(f"PID Cost: {pid_cost['total_cost']:.2f}")
    print(f"  - lataccel: {pid_cost['lataccel_cost']:.2f}")
    print(f"  - jerk: {pid_cost['jerk_cost']:.2f}")
    
    # Run BC
    print("\n[2/2] Running BC...")
    model = TinyPhysicsModel(model_path, debug=False)
    bc = BCController()
    sim_bc = TinyPhysicsSimulator(model, file_path, controller=bc, debug=False)
    sim_bc.rollout()
    bc_cost = sim_bc.compute_cost()
    
    print(f"BC Cost: {bc_cost['total_cost']:.2f}")
    print(f"  - lataccel: {bc_cost['lataccel_cost']:.2f}")
    print(f"  - jerk: {bc_cost['jerk_cost']:.2f}")
    
    print(f"\nBC is {bc_cost['total_cost'] / pid_cost['total_cost']:.1f}× worse than PID")
    
    # Compare actions
    print("\n" + "="*60)
    print("ACTION COMPARISON")
    print("="*60)
    
    pid_actions = np.array(sim_pid.action_history[100:200])
    bc_actions = np.array(sim_bc.action_history[100:200])
    
    print(f"\nPID actions (first 100 control steps):")
    print(f"  Mean: {pid_actions.mean():.4f}")
    print(f"  Std:  {pid_actions.std():.4f}")
    print(f"  Min:  {pid_actions.min():.4f}")
    print(f"  Max:  {pid_actions.max():.4f}")
    
    print(f"\nBC actions (first 100 control steps):")
    print(f"  Mean: {bc_actions.mean():.4f}")
    print(f"  Std:  {bc_actions.std():.4f}")
    print(f"  Min:  {bc_actions.min():.4f}")
    print(f"  Max:  {bc_actions.max():.4f}")
    
    print(f"\nAction MAE: {np.abs(pid_actions - bc_actions).mean():.4f}")
    print(f"Action correlation: {np.corrcoef(pid_actions, bc_actions)[0,1]:.4f}")
    
    # Check if BC is just outputting constant
    if bc_actions.std() < 0.01:
        print("\n⚠️  WARNING: BC actions have very low variance!")
        print("    Network might be outputting constant value")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(pid_actions, label='PID', alpha=0.7)
    plt.plot(bc_actions, label='BC', alpha=0.7)
    plt.ylabel('Steering Action')
    plt.legend()
    plt.title('Actions Comparison (steps 100-200)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(sim_pid.current_lataccel_history[100:200], label='PID', alpha=0.7)
    plt.plot(sim_bc.current_lataccel_history[100:200], label='BC', alpha=0.7)
    plt.plot(sim_pid.target_lataccel_history[100:200], 'k--', label='Target', alpha=0.5)
    plt.ylabel('Lateral Accel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    pid_error = np.array(sim_pid.target_lataccel_history[100:200]) - np.array(sim_pid.current_lataccel_history[100:200])
    bc_error = np.array(sim_bc.target_lataccel_history[100:200]) - np.array(sim_bc.current_lataccel_history[100:200])
    plt.plot(pid_error, label='PID error', alpha=0.7)
    plt.plot(bc_error, label='BC error', alpha=0.7)
    plt.ylabel('Tracking Error')
    plt.xlabel('Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/debug_comparison.png', dpi=150)
    print(f"\n✅ Saved plot to results/debug_comparison.png")
    
    # Test one state
    print("\n" + "="*60)
    print("SINGLE STATE TEST")
    print("="*60)
    
    from controllers.bc import build_state
    state, target, future_plan = sim_bc.get_state_target_futureplan(150)
    current_lataccel = sim_bc.current_lataccel_history[150]
    
    state_vec = build_state(target, current_lataccel, state, future_plan)
    print(f"\nState vector shape: {state_vec.shape}")
    print(f"State values:")
    print(f"  error: {state_vec[0]:.4f}")
    print(f"  roll_lataccel: {state_vec[1]:.4f}")
    print(f"  v_ego: {state_vec[2]:.4f}")
    print(f"  a_ego: {state_vec[3]:.4f}")
    print(f"  current_lataccel: {state_vec[4]:.4f}")
    print(f"  future_lataccel[0]: {state_vec[5]:.4f}")
    print(f"  future_lataccel[-1]: {state_vec[-1]:.4f}")
    print(f"  future mean: {state_vec[5:].mean():.4f}")
    
    # Check for NaN or inf
    if np.any(np.isnan(state_vec)) or np.any(np.isinf(state_vec)):
        print("\n⚠️  WARNING: State contains NaN or Inf!")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()

