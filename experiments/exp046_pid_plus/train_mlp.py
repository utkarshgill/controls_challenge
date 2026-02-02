"""
Train 2-layer MLP
Architecture: features → [16 hidden] → tanh → [1 output]

This can learn feature interactions like:
"if derivative is high AND error is low → sharp turn coming"
"""

import numpy as np
import pickle
from pathlib import Path


def forward(X, W1, b1, W2, b2):
    """Forward pass"""
    h = np.tanh(X @ W1 + b1)  # Hidden layer
    y = h @ W2 + b2            # Output layer
    return y, h


def train_mlp(X, y, hidden_size=16, lr=0.01, epochs=2000):
    """Train with gradient descent"""
    n_features = X.shape[1]
    
    # Initialize weights (Xavier initialization)
    W1 = np.random.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(1)
    
    for epoch in range(epochs):
        # Forward
        y_pred, h = forward(X, W1, b1, W2, b2)
        y_pred = y_pred.flatten()
        loss = np.mean((y - y_pred) ** 2)
        
        # Backward
        dloss = 2 * (y_pred - y).reshape(-1, 1) / len(y)
        
        # Output layer gradients
        dW2 = h.T @ dloss
        db2 = np.sum(dloss, axis=0)
        
        # Hidden layer gradients
        dh = dloss @ W2.T
        dtanh = 1 - h ** 2
        dh_pre = dh * dtanh
        
        dW1 = X.T @ dh_pre
        db1 = np.sum(dh_pre, axis=0)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss:.6f}")
    
    final_loss = np.mean((y - forward(X, W1, b1, W2, b2)[0].flatten()) ** 2)
    print(f"  Final loss: {final_loss:.6f}")
    
    return W1, b1, W2, b2


if __name__ == '__main__':
    # Load data
    data_file = Path(__file__).parent / 'training_data.pkl'
    print("Loading training data...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    X = np.array([d['features'] for d in data])
    y = np.array([d['ff_action'] for d in data])
    
    print(f"Loaded {len(data)} samples\n")
    
    # Train MLP
    print("Training 2-layer MLP (hidden_size=16)...")
    W1, b1, W2, b2 = train_mlp(X, y, hidden_size=16)
    
    # Evaluate
    y_pred, _ = forward(X, W1, b1, W2, b2)
    y_pred = y_pred.flatten()
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    
    print(f"\nTraining performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²:  {r2:.4f}")
    
    # Save
    output_file = Path(__file__).parent / 'mlp_weights.npz'
    np.savez(output_file, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"\nSaved to: {output_file}")
    
    print("\nMLP can learn:")
    print("  - Feature interactions (AND/OR logic)")
    print("  - Nonlinear responses to extreme values")
    print("  - Context-dependent strategies")
