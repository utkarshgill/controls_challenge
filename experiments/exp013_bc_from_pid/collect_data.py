"""
Collect PID trajectories for behavioral cloning
CRITICAL: Only collect samples from step >= CONTROL_START_IDX (100)
Steps 0-99 have logged steerCommand values, NOT PID actions!
"""
import sys
sys.path.insert(0, '../..')

import numpy as np
import pickle
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX, FUTURE_PLAN_STEPS
from controllers.pid import Controller as PIDController


def collect_from_file(data_file):
    """
    Collect (state, action) pairs from a single file
    
    State: [error, error_integral, v_ego, a_ego, roll_lataccel, curvatures[50]]
    Action: PID steer command
    
    ONLY collects from step >= CONTROL_START_IDX (100) to avoid logged steer commands!
    """
    model = TinyPhysicsModel("../../models/tinyphysics.onnx", debug=False)
    controller = PIDController()
    sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
    
    states = []
    actions = []
    
    # Run full rollout
    for step_idx in range(20, len(sim.data)):  # Start from CONTEXT_LENGTH (20)
        # CRITICAL CHECK: Only collect if step_idx >= CONTROL_START_IDX
        if step_idx < CONTROL_START_IDX:
            sim.step()
            continue
        
        # Need enough future steps for curvature calculation
        if step_idx + FUTURE_PLAN_STEPS >= len(sim.data):
            break
        
        # Get state BEFORE controller acts
        state_obj, target_lataccel, future_plan = sim.get_state_target_futureplan(step_idx)
        
        # Calculate error and error_integral
        error = target_lataccel - sim.current_lataccel
        error_integral = controller.error_integral
        
        # Extract state components
        v_ego = state_obj.v_ego
        a_ego = state_obj.a_ego
        roll_lataccel = state_obj.roll_lataccel
        
        # Calculate curvatures from future plan
        # curvature = (lat_accel - roll) / v²
        future_lataccels = np.array(future_plan.lataccel)
        future_rolls = np.array(future_plan.roll_lataccel)
        future_v_egos = np.array(future_plan.v_ego)
        
        # Avoid division by zero (use 1.0 to match controller)
        v_ego_sq = np.maximum(future_v_egos ** 2, 1.0)
        curvatures = (future_lataccels - future_rolls) / v_ego_sq
        
        # Pad if needed (use 'constant' for empty, 'edge' otherwise to match controller)
        if len(curvatures) < FUTURE_PLAN_STEPS:
            pad_mode = 'constant' if len(curvatures) == 0 else 'edge'
            curvatures = np.pad(curvatures, (0, FUTURE_PLAN_STEPS - len(curvatures)), mode=pad_mode)
        
        # Build state vector [error, error_integral, v_ego, a_ego, roll_lataccel, curvatures[50]]
        state = np.concatenate([
            [error, error_integral, v_ego, a_ego, roll_lataccel],
            curvatures[:FUTURE_PLAN_STEPS]
        ])
        
        # Get action from PID controller
        action = controller.update(target_lataccel, sim.current_lataccel, state_obj, future_plan)
        
        states.append(state)
        actions.append(action)
        
        # Step the simulator
        sim.step()
    
    return np.array(states), np.array(actions)


def main():
    print("="*60)
    print("Collecting PID trajectories (FIXED VERSION)")
    print("="*60)
    
    # Get all data files
    data_dir = Path("../../data")
    all_files = sorted(list(data_dir.glob("*.csv")))
    
    print(f"\nTotal files available: {len(all_files)}")
    
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
    
    print(f"\nState statistics:")
    print(f"  Min: {all_states.min(axis=0)[:5]}... (first 5 dims)")
    print(f"  Max: {all_states.max(axis=0)[:5]}... (first 5 dims)")
    print(f"  Mean: {all_states.mean(axis=0)[:5]}... (first 5 dims)")
    
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
    
    output_file = 'pid_trajectories_v3.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"\n✅ Saved to: {output_file}")
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Verify CONTROL_START_IDX filtering
    print(f"\n" + "="*60)
    print(f"VERIFICATION:")
    print(f"  Expected samples/file (if 1000 steps): {1000 - CONTROL_START_IDX - FUTURE_PLAN_STEPS} = {1000 - CONTROL_START_IDX - FUTURE_PLAN_STEPS}")
    print(f"  Actual samples/file (avg): {len(all_states) / len(train_files):.1f}")
    print(f"  CONTROL_START_IDX: {CONTROL_START_IDX}")
    print(f"  This confirms we only collected from step >= {CONTROL_START_IDX}")
    print(f"="*60 + "\n")


if __name__ == '__main__':
    main()
