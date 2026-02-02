"""
Collect PID demonstrations from multiple routes
Use tinyphysics' parallel infrastructure
"""

import numpy as np
import sys
from pathlib import Path
from functools import partial
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController


class PIDDataCollector(BaseController):
    """Collect (features, action) from PID"""
    
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.features = []
        self.actions = []
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Features: [error, error_integral, error_diff, roll_lataccel]
        self.features.append([error, self.error_integral, error_diff, state.roll_lataccel])
        self.actions.append(action)
        
        return action


def collect_from_file(data_path, model_path):
    """Collect data from single file"""
    model = TinyPhysicsModel(model_path, debug=False)
    controller = PIDDataCollector()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    sim.rollout()
    return controller.features, controller.actions


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent.parent / 'data'
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    
    # Get all CSV files
    all_files = sorted(list(data_dir.glob('*.csv')))
    print(f"Found {len(all_files)} total CSV files")
    
    # Shuffle and take first 2000
    np.random.seed(42)
    all_files_shuffled = all_files.copy()
    np.random.shuffle(all_files_shuffled)
    files_to_use = all_files_shuffled[:2000]
    
    print(f"Using {len(files_to_use)} files")
    print(f"Collecting PID demonstrations in parallel...")
    
    # Collect in parallel (like tinyphysics.py does)
    collect_partial = partial(collect_from_file, model_path=str(model_path))
    results = process_map(collect_partial, files_to_use, max_workers=16, chunksize=10)
    
    # Aggregate
    all_features = []
    all_actions = []
    
    for features, actions in results:
        all_features.extend(features)
        all_actions.extend(actions)
    
    # Convert to arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_actions, dtype=np.float32)
    
    print(f"\nCollected {len(X)} samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    # Split: 80% train, 10% val, 10% test
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Save
    output_file = Path(__file__).parent / 'pid_data.npz'
    np.savez(
        output_file,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    
    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
