"""
Experiment 47: Neural Atom
Step 3: Add activation function

Linear worked: R²=1.0, exact PID
Now add tanh: Can it still learn to behave like PID?

Network: features → tanh → output
(Weights will be different, but behavior should match)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController, STEER_RANGE


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
        
        self.features.append([error, self.error_integral, error_diff])
        self.actions.append(action)
        
        return action


def train_with_activation(X, y, lr=0.01, epochs=1000):
    """Train: output = tanh(w · features + b)"""
    
    # Initialize
    w = np.random.randn(3) * 0.1
    b = 0.0
    
    print(f"Training with tanh activation...")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    
    for epoch in range(epochs):
        # Forward
        z = X @ w + b
        y_pred = np.tanh(z)
        loss = np.mean((y - y_pred) ** 2)
        
        # Backward
        dloss = 2 * (y_pred - y) / len(y)
        dtanh = 1 - y_pred ** 2
        dz = dloss * dtanh
        
        dw = X.T @ dz
        db = np.sum(dz)
        
        # Update
        w -= lr * dw
        b -= lr * db
        
        if epoch % 200 == 0:
            print(f"    Epoch {epoch:4d}: loss = {loss:.6f}")
    
    final_loss = np.mean((y - np.tanh(X @ w + b)) ** 2)
    print(f"    Final loss: {final_loss:.6f}")
    
    return w, b


if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("STEP 3: Add tanh activation")
    print("="*60)
    
    # Collect PID data
    print("\nCollecting PID data...")
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = PIDDataCollector()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost_pid = sim.rollout()
    
    print(f"  PID cost: {cost_pid['total_cost']:.2f}")
    print(f"  Samples: {len(controller.features)}")
    
    # Train with activation
    X = np.array(controller.features)
    y = np.array(controller.actions)
    
    print(f"\nTraining nonlinear neuron...")
    w, b = train_with_activation(X, y)
    
    print(f"\n{'='*60}")
    print(f"LEARNED PARAMETERS:")
    print(f"{'='*60}")
    print(f"Weights: {w}")
    print(f"Bias:    {b:.6f}")
    
    # Check fit
    y_pred = np.tanh(X @ w + b)
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    print(f"\nFit quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²:  {r2:.6f}")
    
    # Save for controller
    params = {'w': w, 'b': b}
    params_file = Path(__file__).parent / 'tanh_params.npz'
    np.savez(params_file, **params)
    print(f"\nSaved parameters to: {params_file}")
    
    print(f"\n{'='*60}")
    print(f"NEXT: Create controller and run batch eval")
    print(f"{'='*60}")
    print(f"\nExpected:")
    print(f"  PID (batch): 84.85")
    print(f"  This neuron: ~85 (should match)")
