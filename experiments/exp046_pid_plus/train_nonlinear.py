"""
Train single neuron with tanh activation
Learn: ff_action = tanh(w · features + b)
"""

import numpy as np
import pickle
from pathlib import Path


def model(params, X):
    """Single neuron with tanh"""
    w = params[:-1]
    b = params[-1]
    return np.tanh(X @ w + b)


def loss_and_grad(params, X, y):
    """MSE loss and gradient"""
    w = params[:-1]
    b = params[-1]
    
    # Forward
    z = X @ w + b
    y_pred = np.tanh(z)
    loss = np.mean((y - y_pred) ** 2)
    
    # Backward
    dloss = 2 * (y_pred - y) / len(y)
    dtanh = 1 - y_pred ** 2  # tanh gradient
    dz = dloss * dtanh
    
    dw = X.T @ dz
    db = np.sum(dz)
    
    grad = np.concatenate([dw, [db]])
    return loss, grad


if __name__ == '__main__':
    # Load data
    data_file = Path(__file__).parent / 'training_data.pkl'
    print("Loading training data...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    X = np.array([d['features'] for d in data])
    y = np.array([d['ff_action'] for d in data])
    
    print(f"Loaded {len(data)} samples\n")
    
    # Initialize with linear weights
    linear_weights = np.load(Path(__file__).parent / 'linear_weights.npy')
    params = np.concatenate([linear_weights, [0.0]])  # Add bias
    
    print("Training nonlinear model (tanh)...")
    
    # Gradient descent
    lr = 0.01
    for epoch in range(1000):
        loss_val, grad = loss_and_grad(params, X, y)
        params = params - lr * grad
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss_val:.6f}")
    
    final_loss, _ = loss_and_grad(params, X, y)
    print(f"  Final loss: {final_loss:.6f}\n")
    w = params[:-1]
    b = params[-1]
    
    print("Learned weights:")
    feature_names = ['immediate_err', 'derivative', 'current_err', 'prev_action', 'v_ego', 'roll']
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: {w[i]:8.4f}")
    print(f"  bias           : {b:8.4f}")
    
    # Evaluate
    y_pred = model(params, X)
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    
    print(f"\nTraining performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²:  {r2:.4f}")
    
    # Save
    output_file = Path(__file__).parent / 'nonlinear_weights.npy'
    np.save(output_file, params)
    print(f"\nSaved to: {output_file}")
