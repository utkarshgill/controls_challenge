"""
Collect PID trajectories with CURVATURE-SPACE representation
State: 58D including current_curvature, target_curvature, error_derivative, friction_available
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pickle
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, FUTURE_PLAN_STEPS, DEL_T
from controllers.pid import Controller as PIDController


def collect_from_file(data_file):
    """
    Collect (state, action) pairs with curvature-space representation
    
    State (58D): [current_curvature, target_curvature, curvature_error, 
                  curvature_error_integral, curvature_error_derivative,
                  v_ego, a_ego, friction_available, future_curvatures[50]]
    Action: PID steer command
    """
    model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
    controller = PIDController()
    sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
    
    states = []
    actions = []
    
    # Track previous curvature error for derivative
    prev_curvature_error = 0.0
    curvature_error_integral = 0.0
    
    # Run full rollout
    for step_idx in range(20, len(sim.data)):
        # CRITICAL: Only collect if step_idx >= CONTROL_START_IDX
        if step_idx < CONTROL_START_IDX:
            sim.step()
            continue
        
        # Need enough future steps
        if step_idx + FUTURE_PLAN_STEPS >= len(sim.data):
            break
        
        # Get state BEFORE controller acts
        state_obj, target_lataccel, future_plan = sim.get_state_target_futureplan(step_idx)
        
        # Current vehicle state
        v_ego = state_obj.v_ego
        a_ego = state_obj.a_ego
        roll_lataccel = state_obj.roll_lataccel
        current_lataccel = sim.current_lataccel
        
        # Convert to curvature space (speed-invariant)
        v_ego_sq = max(v_ego ** 2, 1.0)
        current_curvature = (current_lataccel - roll_lataccel) / v_ego_sq  # Subtract roll (consistent with future)
        target_curvature = target_lataccel / v_ego_sq
        
        # Curvature error (replaces lataccel error)
        curvature_error = target_curvature - current_curvature
        curvature_error_integral += curvature_error
        curvature_error_derivative = (curvature_error - prev_curvature_error) / DEL_T
        prev_curvature_error = curvature_error
        
        # Friction circle: available lateral capacity
        # Total grip ~10 m/s² (assumed), shared between lateral and longitudinal
        max_combined_accel = 10.0
        longitudinal_fraction = min(abs(a_ego) / max_combined_accel, 1.0)
        friction_available = np.sqrt(max(0.0, 1.0 - longitudinal_fraction**2))
        
        # Calculate future curvatures
        future_lataccels = np.array(future_plan.lataccel)
        future_rolls = np.array(future_plan.roll_lataccel)
        future_v_egos = np.array(future_plan.v_ego)
        
        # Each timestep uses its own future speed
        v_ego_sq_future = np.maximum(future_v_egos ** 2, 1.0)
        curvatures = (future_lataccels - future_rolls) / v_ego_sq_future
        
        # Pad if needed
        if len(curvatures) < FUTURE_PLAN_STEPS:
            pad_mode = 'constant' if len(curvatures) == 0 else 'edge'
            curvatures = np.pad(curvatures, (0, FUTURE_PLAN_STEPS - len(curvatures)), mode=pad_mode)
        
        # Build state vector (58D)
        state = np.concatenate([
            [current_curvature, target_curvature, curvature_error, 
             curvature_error_integral, curvature_error_derivative,
             v_ego, a_ego, friction_available],
            curvatures[:FUTURE_PLAN_STEPS]
        ])
        
        # Get action from PID controller (still uses lataccel space internally)
        action = controller.update(target_lataccel, current_lataccel, state_obj, future_plan)
        
        states.append(state)
        actions.append(action)
        
        # Step the simulator
        sim.step()
    
    return np.array(states), np.array(actions)


def main():
    print("="*60)
    print("Collecting PID trajectories (CURVATURE SPACE)")
    print("="*60)
    print("\nNew state representation:")
    print("  - Current & target curvatures (speed-invariant)")
    print("  - Error derivative (PID D-term)")
    print("  - Friction circle (available grip)")
    print("  - 58D total (vs 55D before)")
    print()
    
    # Get all data files
    data_dir = Path("../../data")
    all_files = sorted(list(data_dir.glob("*.csv")))
    
    print(f"Total files available: {len(all_files)}")
    
    # Sample 5000 random files for training
    np.random.seed(42)
    train_indices = np.random.choice(len(all_files), size=5000, replace=False)
    train_files = [all_files[i] for i in train_indices]
    
    print(f"Collecting from: {len(train_files)} files")
    print(f"Using {16} parallel workers")
    
    # Collect data in parallel
    results = process_map(collect_from_file, train_files, max_workers=16, chunksize=1)
    
    # Aggregate results
    all_states = []
    all_actions = []
    
    for states, actions in results:
        if len(states) > 0:
            all_states.append(states)
            all_actions.append(actions)
    
    all_states = np.vstack(all_states)
    all_actions = np.concatenate(all_actions)
    
    print(f"\n" + "="*60)
    print(f"Data collection complete!")
    print(f"="*60)
    print(f"Total samples: {len(all_states):,}")
    print(f"State shape: {all_states.shape}")
    print(f"Action shape: {all_actions.shape}")
    
    print(f"\nState statistics (first 8 dims):")
    print(f"  [0] current_curvature:     mean={all_states[:, 0].mean():.6f}, std={all_states[:, 0].std():.6f}")
    print(f"  [1] target_curvature:      mean={all_states[:, 1].mean():.6f}, std={all_states[:, 1].std():.6f}")
    print(f"  [2] curvature_error:       mean={all_states[:, 2].mean():.6f}, std={all_states[:, 2].std():.6f}")
    print(f"  [3] curvature_error_int:   mean={all_states[:, 3].mean():.6f}, std={all_states[:, 3].std():.6f}")
    print(f"  [4] curvature_error_deriv: mean={all_states[:, 4].mean():.6f}, std={all_states[:, 4].std():.6f}")
    print(f"  [5] v_ego:                 mean={all_states[:, 5].mean():.2f}, std={all_states[:, 5].std():.2f}")
    print(f"  [6] a_ego:                 mean={all_states[:, 6].mean():.2f}, std={all_states[:, 6].std():.2f}")
    print(f"  [7] friction_available:    mean={all_states[:, 7].mean():.4f}, std={all_states[:, 7].std():.4f}")
    
    print(f"\nAction statistics:")
    print(f"  Min: {all_actions.min():.4f}")
    print(f"  Max: {all_actions.max():.4f}")
    print(f"  Mean: {all_actions.mean():.4f}")
    print(f"  Std: {all_actions.std():.4f}")
    
    # Save to disk
    data = {
        'states': all_states,
        'actions': all_actions,
        'train_indices': train_indices
    }
    
    output_file = 'pid_trajectories_curvature.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved to: {output_file}")
    print(f"File size: {file_size_mb:.1f} MB")
    
    print(f"\n" + "="*60)
    print(f"Ready to train with curvature-space representation!")
    print(f"="*60 + "\n")


if __name__ == '__main__':
    main()

