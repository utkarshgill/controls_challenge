"""
Train single linear neuron
Learn: ff_action = w · features
"""

import numpy as np
import pickle
from pathlib import Path


if __name__ == '__main__':
    # Load data
    data_file = Path(__file__).parent / 'training_data.pkl'
    print("Loading training data...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples\n")
    
    # Prepare X, y
    X = np.array([d['features'] for d in data])  # (N, 6)
    y = np.array([d['ff_action'] for d in data])  # (N,)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}\n")
    
    # Linear regression: w = (X^T X)^-1 X^T y
    print("Training linear model...")
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    
    print("Learned weights:")
    feature_names = ['immediate_err', 'derivative', 'current_err', 'prev_action', 'v_ego', 'roll']
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: {w[i]:8.4f}")
    
    # Evaluate on training data
    y_pred = X @ w
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    
    print(f"\nTraining performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²:  {r2:.4f}")
    
    # Save weights
    weights_file = Path(__file__).parent / 'linear_weights.npy'
    np.save(weights_file, w)
    print(f"\nSaved weights to: {weights_file}")
    
    # Compare to hand-crafted v3
    print(f"\nComparison to v3 (hand-crafted):")
    print(f"  v3 immediate: 0.30 → learned: {w[0]:.4f}")
    print(f"  v3 derivative: 0.15 → learned: {w[1]:.4f}")
