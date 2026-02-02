"""
Step 4: Learn to use ALL 50 future lataccels
Single neuron with 53 weights: [error, error_integral, error_diff, lataccel[0..49]]
"""

import numpy as np
import sys
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController, FUTURE_PLAN_STEPS


class BestFFController(BaseController):
    """exp046_v3 - best performing controller (82.89)"""
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Feedforward params
        self.ff_immediate = 0.3  
        self.ff_derivative = 0.15
        
        # For data collection
        self.features = []
        self.actions = []
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Feedforward
        ff_action = 0.0
        if len(future_plan.lataccel) >= 5:
            immediate_error = future_plan.lataccel[0] - target_lataccel
            ff_action += self.ff_immediate * immediate_error
            
            future_slice = future_plan.lataccel[:5]
            target_derivative = (future_slice[-1] - future_slice[0]) / 5.0
            ff_action += self.ff_derivative * target_derivative
        
        # Collect features: (future_lataccel[i] - target_lataccel) for i in [0..49]
        # This represents how the future trajectory differs from current target
        future_padded = list(future_plan.lataccel[:FUTURE_PLAN_STEPS]) + [target_lataccel] * (FUTURE_PLAN_STEPS - len(future_plan.lataccel))
        features = [(f - target_lataccel) for f in future_padded[:FUTURE_PLAN_STEPS]]
        
        self.features.append(features)
        self.actions.append(ff_action)  # Only FF action (PID is separate)
        
        return pid_action + ff_action


def collect_from_file(data_path, model_path):
    """Collect data from single file"""
    model = TinyPhysicsModel(model_path, debug=False)
    controller = BestFFController()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    sim.rollout()
    return controller.features, controller.actions


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent.parent / 'data'
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    
    # Use same 2000 routes
    all_files = sorted(list(data_dir.glob('*.csv')))
    np.random.seed(42)
    all_files_shuffled = all_files.copy()
    np.random.shuffle(all_files_shuffled)
    files_to_use = all_files_shuffled[:2000]
    
    print("="*60)
    print("STEP 4: Collect data with ALL 50 future lataccels")
    print("="*60)
    print(f"Using {len(files_to_use)} files")
    print(f"Features: (future_lataccel[i] - current_target) for i=0..49")
    print(f"Target: FF action from best controller (exp046_v3)")
    print(f"PID is fixed (0.195, 0.100, -0.053)")
    print(f"Hypothesis: Weights should decay with distance")
    print()
    
    # Collect in parallel
    collect_partial = partial(collect_from_file, model_path=str(model_path))
    results = process_map(collect_partial, files_to_use, max_workers=16, chunksize=10)
    
    # Aggregate
    all_features = []
    all_actions = []
    for features, actions in results:
        all_features.extend(features)
        all_actions.extend(actions)
    
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_actions, dtype=np.float32)
    
    print(f"Collected {len(X):,} samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    # Split: 80/10/10
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"Split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print()
    
    # Train single linear neuron: w · features = action
    print("Training linear neuron...")
    weights, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
    
    # Evaluate
    y_train_pred = X_train @ weights
    y_val_pred = X_val @ weights
    y_test_pred = X_test @ weights
    
    train_r2 = 1 - np.mean((y_train - y_train_pred)**2) / np.var(y_train)
    val_r2 = 1 - np.mean((y_val - y_val_pred)**2) / np.var(y_val)
    test_r2 = 1 - np.mean((y_test - y_test_pred)**2) / np.var(y_test)
    
    print(f"Results:")
    print(f"  Train R²: {train_r2:.6f}")
    print(f"  Val R²:   {val_r2:.6f}")
    print(f"  Test R²:  {test_r2:.6f}")
    print()
    
    # Analyze weights
    print("Future lataccel weights (should decay with distance):")
    for i in [0, 1, 2, 5, 10, 20, 30, 40, 49]:
        print(f"  w[{i}] (lataccel[{i}], {i*0.1:.1f}s ahead): {weights[i]:+.6f}")
    print()
    
    # Save
    output_file = Path(__file__).parent / 'full_future_weights.npy'
    np.save(output_file, weights)
    print(f"Saved to: {output_file}")
    print()
    print("="*60)
    print("NEXT: Deploy controller and run batch eval")
    print("Expected: Better than 82.89 (exp046_v3)")
    print("="*60)
