"""
Experiment 47: Neural Atom
Step 1: Can a single neuron learn to be PID?

Input: [error, error_integral, error_diff]
Output: action = w0*error + w1*error_integral + w2*error_diff

Target: w = [0.195, 0.100, -0.053] (PID coefficients)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController


class PIDDataCollector(BaseController):
    """Collect (features, action) pairs from PID"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        
        # State
        self.error_integral = 0
        self.prev_error = 0
        
        # Data
        self.features = []
        self.actions = []
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute features
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # Compute PID action
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Store
        self.features.append([error, self.error_integral, error_diff])
        self.actions.append(action)
        
        return action


if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("STEP 1: Can single neuron learn PID?")
    print("="*60)
    
    # Collect PID data
    print("\nCollecting PID data from single trajectory...")
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = PIDDataCollector()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    
    print(f"  PID cost: {cost['total_cost']:.2f}")
    print(f"  Samples collected: {len(controller.features)}")
    
    # Prepare data
    X = np.array(controller.features)  # (N, 3)
    y = np.array(controller.actions)    # (N,)
    
    print(f"\nData shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    
    # Train: w = (X^T X)^-1 X^T y
    print(f"\nTraining linear neuron...")
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"\nTarget (PID coefficients):")
    print(f"  error:          {controller.p:.6f}")
    print(f"  error_integral: {controller.i:.6f}")
    print(f"  error_diff:     {controller.d:.6f}")
    
    print(f"\nLearned weights:")
    print(f"  error:          {w[0]:.6f}")
    print(f"  error_integral: {w[1]:.6f}")
    print(f"  error_diff:     {w[2]:.6f}")
    
    print(f"\nErrors:")
    print(f"  error:          {abs(w[0] - controller.p):.6f}")
    print(f"  error_integral: {abs(w[1] - controller.i):.6f}")
    print(f"  error_diff:     {abs(w[2] - controller.d):.6f}")
    print(f"  Total error:    {np.sum(np.abs(w - [controller.p, controller.i, controller.d])):.6f}")
    
    # Verify on data
    y_pred = X @ w
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    
    print(f"\nFit quality:")
    print(f"  MSE: {mse:.9f}")
    print(f"  R²:  {r2:.9f}")
    
    if r2 > 0.9999:
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: Single neuron EXACTLY recovers PID!")
        print(f"{'='*60}")
        
        # Save weights for step 2
        weights_file = Path(__file__).parent / 'pid_weights.npy'
        np.save(weights_file, w)
        print(f"\nSaved weights to: {weights_file}")
        print(f"\nReady for Step 2: Add state features")
    else:
        print(f"\n✗ FAILED: R² = {r2:.6f} (expected > 0.9999)")
