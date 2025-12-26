#!/usr/bin/env python3
"""Deep debugging: PPO vs PID"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import matplotlib.pyplot as plt
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers.pid import Controller as PIDController
from controllers.ppo import Controller as PPOController

def analyze_single_file(file_path, model_path):
    """Deep analysis on one file"""
    print("="*80)
    print(f"ANALYZING: {file_path}")
    print("="*80)
    
    # Run PID
    print("\n[1/2] Running PID...")
    model = TinyPhysicsModel(model_path, debug=False)
    pid = PIDController()
    sim_pid = TinyPhysicsSimulator(model, file_path, controller=pid, debug=False)
    sim_pid.rollout()
    pid_cost = sim_pid.compute_cost()
    
    # Run PPO
    print("[2/2] Running PPO...")
    model = TinyPhysicsModel(model_path, debug=False)
    ppo = PPOController()
    sim_ppo = TinyPhysicsSimulator(model, file_path, controller=ppo, debug=False)
    sim_ppo.rollout()
    ppo_cost = sim_ppo.compute_cost()
    
    print("\n" + "="*80)
    print("COST COMPARISON")
    print("="*80)
    print(f"\nPID:")
    print(f"  Lataccel: {pid_cost['lataccel_cost']:8.2f}")
    print(f"  Jerk:     {pid_cost['jerk_cost']:8.2f}")
    print(f"  Total:    {pid_cost['total_cost']:8.2f}")
    
    print(f"\nPPO:")
    print(f"  Lataccel: {ppo_cost['lataccel_cost']:8.2f}")
    print(f"  Jerk:     {ppo_cost['jerk_cost']:8.2f}")
    print(f"  Total:    {ppo_cost['total_cost']:8.2f}")
    
    print(f"\nRatio: PPO is {ppo_cost['total_cost'] / pid_cost['total_cost']:.2f}× worse")
    
    # Action analysis
    print("\n" + "="*80)
    print("ACTION STATISTICS (control steps 100-500)")
    print("="*80)
    
    pid_actions = np.array(sim_pid.action_history[100:500])
    ppo_actions = np.array(sim_ppo.action_history[100:500])
    
    print(f"\nPID actions:")
    print(f"  Mean:  {pid_actions.mean():7.4f}")
    print(f"  Std:   {pid_actions.std():7.4f}")
    print(f"  Min:   {pid_actions.min():7.4f}")
    print(f"  Max:   {pid_actions.max():7.4f}")
    print(f"  Range: {pid_actions.max() - pid_actions.min():7.4f}")
    
    print(f"\nPPO actions:")
    print(f"  Mean:  {ppo_actions.mean():7.4f}")
    print(f"  Std:   {ppo_actions.std():7.4f}")
    print(f"  Min:   {ppo_actions.min():7.4f}")
    print(f"  Max:   {ppo_actions.max():7.4f}")
    print(f"  Range: {ppo_actions.max() - ppo_actions.min():7.4f}")
    
    print(f"\nAction comparison:")
    print(f"  MAE:         {np.abs(pid_actions - ppo_actions).mean():7.4f}")
    print(f"  RMSE:        {np.sqrt(((pid_actions - ppo_actions)**2).mean()):7.4f}")
    print(f"  Correlation: {np.corrcoef(pid_actions, ppo_actions)[0,1]:7.4f}")
    
    # Tracking error analysis
    print("\n" + "="*80)
    print("TRACKING ERROR (control steps 100-500)")
    print("="*80)
    
    pid_error = np.array(sim_pid.target_lataccel_history[100:500]) - np.array(sim_pid.current_lataccel_history[100:500])
    ppo_error = np.array(sim_ppo.target_lataccel_history[100:500]) - np.array(sim_ppo.current_lataccel_history[100:500])
    
    print(f"\nPID tracking error:")
    print(f"  Mean abs: {np.abs(pid_error).mean():7.4f}")
    print(f"  RMS:      {np.sqrt((pid_error**2).mean()):7.4f}")
    print(f"  Max abs:  {np.abs(pid_error).max():7.4f}")
    
    print(f"\nPPO tracking error:")
    print(f"  Mean abs: {np.abs(ppo_error).mean():7.4f}")
    print(f"  RMS:      {np.sqrt((ppo_error**2).mean()):7.4f}")
    print(f"  Max abs:  {np.abs(ppo_error).max():7.4f}")
    
    # Jerk analysis
    print("\n" + "="*80)
    print("JERK ANALYSIS")
    print("="*80)
    
    pid_jerk = np.diff(sim_pid.current_lataccel_history[100:500]) / 0.1
    ppo_jerk = np.diff(sim_ppo.current_lataccel_history[100:500]) / 0.1
    
    print(f"\nPID jerk:")
    print(f"  Mean abs: {np.abs(pid_jerk).mean():7.4f}")
    print(f"  RMS:      {np.sqrt((pid_jerk**2).mean()):7.4f}")
    print(f"  Max abs:  {np.abs(pid_jerk).max():7.4f}")
    
    print(f"\nPPO jerk:")
    print(f"  Mean abs: {np.abs(ppo_jerk).mean():7.4f}")
    print(f"  RMS:      {np.sqrt((ppo_jerk**2).mean()):7.4f}")
    print(f"  Max abs:  {np.abs(ppo_jerk).max():7.4f}")
    
    # Check for oscillations
    print("\n" + "="*80)
    print("OSCILLATION CHECK")
    print("="*80)
    
    pid_sign_changes = np.sum(np.diff(np.sign(pid_actions)) != 0)
    ppo_sign_changes = np.sum(np.diff(np.sign(ppo_actions)) != 0)
    
    print(f"\nPID action sign changes: {pid_sign_changes} / {len(pid_actions)-1} ({100*pid_sign_changes/(len(pid_actions)-1):.1f}%)")
    print(f"PPO action sign changes: {ppo_sign_changes} / {len(ppo_actions)-1} ({100*ppo_sign_changes/(len(ppo_actions)-1):.1f}%)")
    
    if ppo_sign_changes > 2 * pid_sign_changes:
        print("⚠️  PPO is oscillating much more than PID!")
    
    # Plot comparison
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    steps = range(100, 500)
    
    # Actions
    axes[0].plot(steps, pid_actions, label='PID', alpha=0.7, linewidth=1.5)
    axes[0].plot(steps, ppo_actions, label='PPO', alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel('Steering Action')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Actions Comparison')
    
    # Lateral acceleration
    axes[1].plot(steps, sim_pid.target_lataccel_history[100:500], 'k--', label='Target', alpha=0.5, linewidth=1)
    axes[1].plot(steps, sim_pid.current_lataccel_history[100:500], label='PID', alpha=0.7)
    axes[1].plot(steps, sim_ppo.current_lataccel_history[100:500], label='PPO', alpha=0.7)
    axes[1].set_ylabel('Lateral Accel')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Tracking error
    axes[2].plot(steps, pid_error, label='PID error', alpha=0.7)
    axes[2].plot(steps, ppo_error, label='PPO error', alpha=0.7)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Tracking Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Jerk
    axes[3].plot(steps[:-1], pid_jerk, label='PID jerk', alpha=0.7)
    axes[3].plot(steps[:-1], ppo_jerk, label='PPO jerk', alpha=0.7)
    axes[3].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_ylabel('Jerk (m/s³)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Cost accumulation
    pid_cumulative = np.cumsum([(sim_pid.target_lataccel_history[i] - sim_pid.current_lataccel_history[i])**2 * 50 
                                 for i in range(100, 500)])
    ppo_cumulative = np.cumsum([(sim_ppo.target_lataccel_history[i] - sim_ppo.current_lataccel_history[i])**2 * 50
                                 for i in range(100, 500)])
    axes[4].plot(steps, pid_cumulative, label='PID cumulative cost', alpha=0.7)
    axes[4].plot(steps, ppo_cumulative, label='PPO cumulative cost', alpha=0.7)
    axes[4].set_ylabel('Cumulative Cost')
    axes[4].set_xlabel('Step')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/deep_debug.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved plot: results/deep_debug.png")
    
    return pid_cost, ppo_cost

def check_state_normalization():
    """Verify state normalization is working"""
    print("\n" + "="*80)
    print("STATE NORMALIZATION CHECK")
    print("="*80)
    
    from controllers.ppo import build_state, NORM_SCALE
    from tinyphysics import State, FuturePlan
    
    # Test state
    target = 1.5
    current = 1.0
    state = State(roll_lataccel=-0.2, v_ego=30.0, a_ego=-0.5)
    future = FuturePlan(lataccel=[0.5, 1.0, 1.5]*17, roll_lataccel=[0]*50, v_ego=[30]*50, a_ego=[-0.5]*50)
    
    state_vec = build_state(target, current, state, future)
    
    print("\nRaw values:")
    print(f"  error: {target - current:.4f}")
    print(f"  roll:  {state.roll_lataccel:.4f}")
    print(f"  v_ego: {state.v_ego:.4f}")
    print(f"  a_ego: {state.a_ego:.4f}")
    print(f"  current: {current:.4f}")
    
    print("\nNormalized values:")
    print(f"  error: {state_vec[0]:.4f} (raw × {NORM_SCALE[0]:.2f})")
    print(f"  roll:  {state_vec[1]:.4f} (raw × {NORM_SCALE[1]:.2f})")
    print(f"  v_ego: {state_vec[2]:.4f} (raw × {NORM_SCALE[2]:.4f})")
    print(f"  a_ego: {state_vec[3]:.4f} (raw × {NORM_SCALE[3]:.2f})")
    print(f"  current: {state_vec[4]:.4f} (raw × {NORM_SCALE[4]:.2f})")
    print(f"  future[0]: {state_vec[5]:.4f}")
    
    print("\nState statistics:")
    print(f"  Mean: {state_vec.mean():.4f}")
    print(f"  Std:  {state_vec.std():.4f}")
    print(f"  Min:  {state_vec.min():.4f}")
    print(f"  Max:  {state_vec.max():.4f}")
    
    if np.any(np.isnan(state_vec)) or np.any(np.isinf(state_vec)):
        print("\n⚠️  WARNING: State contains NaN or Inf!")
        return False
    
    print("\n✅ State normalization looks OK")
    return True

def main():
    print("\n" + "="*80)
    print("DEEP DEBUGGING: PPO vs PID")
    print("="*80)
    
    # Check state normalization
    check_state_normalization()
    
    # Analyze multiple files
    model_path = "../../models/tinyphysics.onnx"
    test_files = [
        "../../data/00000.csv",
        "../../data/00001.csv",
        "../../data/00002.csv",
    ]
    
    results = []
    for file_path in test_files:
        pid_cost, ppo_cost = analyze_single_file(file_path, model_path)
        results.append((file_path.split('/')[-1], pid_cost, ppo_cost))
        print("\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'File':<15} {'PID Total':<12} {'PPO Total':<12} {'Ratio':<8}")
    print("-"*80)
    for filename, pid, ppo in results:
        ratio = ppo['total_cost'] / pid['total_cost']
        print(f"{filename:<15} {pid['total_cost']:>10.2f}   {ppo['total_cost']:>10.2f}   {ratio:>6.2f}×")
    
    avg_pid = np.mean([r[1]['total_cost'] for r in results])
    avg_ppo = np.mean([r[2]['total_cost'] for r in results])
    print("-"*80)
    print(f"{'Average':<15} {avg_pid:>10.2f}   {avg_ppo:>10.2f}   {avg_ppo/avg_pid:>6.2f}×")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

