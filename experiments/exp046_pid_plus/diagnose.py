"""Diagnose WHERE PID loses points"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER
from controller import Controller

if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("PID DIAGNOSIS")
    print("="*60)
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = Controller()
    
    print("\nRunning PID...")
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    
    # Extract data from control region
    target = np.array(sim.target_lataccel_history)[CONTROL_START_IDX:]
    pred = np.array(sim.current_lataccel_history)[CONTROL_START_IDX:]
    actions = np.array(sim.action_history)[CONTROL_START_IDX:]
    
    # Compute pointwise costs
    error = target - pred
    error_sq = error ** 2
    jerk = np.diff(pred) / DEL_T
    jerk_sq = jerk ** 2
    action_change = np.diff(actions)
    
    print(f"\nTotal cost: {cost['total_cost']:.2f}")
    print(f"  Lat: {cost['lataccel_cost']:.2f} × 50")
    print(f"  Jerk: {cost['jerk_cost']:.2f}")
    
    print(f"\nError stats:")
    print(f"  Mean abs error: {np.mean(np.abs(error)):.4f}")
    print(f"  Max abs error: {np.max(np.abs(error)):.4f}")
    print(f"  Std error: {np.std(error):.4f}")
    
    print(f"\nJerk stats:")
    print(f"  Mean abs jerk: {np.mean(np.abs(jerk)):.4f}")
    print(f"  Max abs jerk: {np.max(np.abs(jerk)):.4f}")
    print(f"  Std jerk: {np.std(jerk):.4f}")
    
    print(f"\nAction stats:")
    print(f"  Mean abs action: {np.mean(np.abs(actions)):.4f}")
    print(f"  Max abs action: {np.max(np.abs(actions)):.4f}")
    print(f"  Mean abs change: {np.mean(np.abs(action_change)):.4f}")
    print(f"  Max abs change: {np.max(np.abs(action_change)):.4f}")
    
    # Find worst regions
    window = 50  # Analyze in 5-second windows
    n_windows = len(error) // window
    
    print(f"\nWorst regions (5s windows):")
    window_costs = []
    for i in range(n_windows):
        start = i * window
        end = start + window
        w_error = error[start:end]
        w_jerk = jerk[start:min(end, len(jerk))]
        w_cost = np.mean(w_error**2) * 100 * LAT_ACCEL_COST_MULTIPLIER + np.mean(w_jerk**2) * 100
        window_costs.append((i, w_cost, np.mean(w_error**2), np.mean(w_jerk**2)))
    
    window_costs.sort(key=lambda x: x[1], reverse=True)
    for i, (win_idx, w_cost, w_err, w_jerk) in enumerate(window_costs[:5]):
        step_start = CONTROL_START_IDX + win_idx * window
        step_end = step_start + window
        print(f"  {i+1}. Steps {step_start}-{step_end}: cost={w_cost:.1f} (err²={w_err:.4f}, jerk²={w_jerk:.4f})")
    
    # Plot
    print("\nGenerating plots...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    steps = np.arange(CONTROL_START_IDX, CONTROL_START_IDX + len(target))
    
    # Tracking
    axes[0].plot(steps, target, 'g-', label='Target', linewidth=2)
    axes[0].plot(steps, pred, 'b-', label='PID', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('Lat Accel')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'PID Performance (Cost={cost["total_cost"]:.1f})')
    
    # Error
    axes[1].plot(steps, error, 'r-', linewidth=1)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Tracking Error')
    
    # Jerk
    axes[2].plot(steps[:-1], jerk, 'orange', linewidth=1)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Jerk (m/s³)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Jerk (rate of change of lat accel)')
    
    # Actions
    axes[3].plot(steps, actions, 'purple', linewidth=1)
    axes[3].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[3].set_ylabel('Steering')
    axes[3].set_xlabel('Step')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Control Actions')
    
    plt.tight_layout()
    plot_path = Path(__file__).parent / 'pid_diagnosis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
