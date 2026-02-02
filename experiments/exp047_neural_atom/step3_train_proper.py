"""
Step 3: Train on proper dataset (2000 routes, 1.16M samples)
"""

import numpy as np
from pathlib import Path


def train_with_activation(X, y, X_val, y_val, lr=0.01, epochs=500):
    """Train: output = tanh(w · features + b)"""
    
    # Initialize
    n_features = X.shape[1]
    w = np.random.randn(n_features) * 0.1
    b = 0.0
    
    best_val_loss = float('inf')
    best_w, best_b = w.copy(), b
    
    print(f"Training:")
    print(f"  Train samples: {len(X):,}")
    print(f"  Val samples:   {len(X_val):,}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}\n")
    
    for epoch in range(epochs):
        # Forward (train)
        z = X @ w + b
        y_pred = np.tanh(z)
        train_loss = np.mean((y - y_pred) ** 2)
        
        # Backward
        dloss = 2 * (y_pred - y) / len(y)
        dtanh = 1 - y_pred ** 2
        dz = dloss * dtanh
        
        dw = X.T @ dz
        db = np.sum(dz)
        
        # Update
        w -= lr * dw
        b -= lr * db
        
        # Validation
        if epoch % 50 == 0:
            z_val = X_val @ w + b
            y_val_pred = np.tanh(z_val)
            val_loss = np.mean((y_val - y_val_pred) ** 2)
            
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_w, best_b = w.copy(), b
    
    print(f"\nBest val loss: {best_val_loss:.6f}")
    
    return best_w, best_b


if __name__ == '__main__':
    # Load data
    data_file = Path(__file__).parent / 'pid_data.npz'
    print("="*60)
    print("STEP 3: Train tanh neuron on proper dataset")
    print("="*60)
    print(f"\nLoading data from {data_file.name}...")
    
    data = np.load(data_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Train
    print(f"\n{'='*60}")
    w, b = train_with_activation(X_train, y_train, X_val, y_val)
    
    # Test
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION:")
    print(f"{'='*60}")
    y_test_pred = np.tanh(X_test @ w + b)
    test_loss = np.mean((y_test - y_test_pred) ** 2)
    test_r2 = 1 - test_loss / np.var(y_test)
    
    print(f"  MSE: {test_loss:.6f}")
    print(f"  R²:  {test_r2:.6f}")
    
    print(f"\nLearned parameters:")
    print(f"  w[0] (error):          {w[0]:.6f}")
    print(f"  w[1] (error_integral): {w[1]:.6f}")
    print(f"  w[2] (error_diff):     {w[2]:.6f}")
    if len(w) > 3:
        print(f"  w[3] (roll_lataccel):  {w[3]:.6f}")
    print(f"  b (bias):              {b:.6f}")
    
    # Save
    params_file = Path(__file__).parent / 'tanh_params_proper.npz'
    np.savez(params_file, w=w, b=b)
    print(f"\nSaved to: {params_file}")
    
    print(f"\n{'='*60}")
    print(f"NEXT: Deploy controller and run batch eval")
    print(f"{'='*60}")
