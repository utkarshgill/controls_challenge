"""
Experiment 016 V3: Analytical Least Squares Solution

Since the data is PERFECTLY linear (verified!), use analytical solution:

weights = (X^T X)^{-1} X^T y

This should recover EXACT PID coefficients with NO optimization error.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from train_v2_stateful import collect_pid_demonstrations_stateful, OneNeuronNetwork
from data_split import get_data_split

def analytical_least_squares(X, y):
    """
    Solve for weights analytically:
    weights = (X^T X)^{-1} X^T y
    
    For Aw = b, the least squares solution is w = (A^T A)^{-1} A^T b
    """
    print("\n" + "="*80)
    print("Analytical Least Squares Solution")
    print("="*80)
    
    # X is (N, 3), y is (N, 1)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Compute X^T X
    XtX = X.T @ X
    print(f"\nX^T X shape: {XtX.shape}")
    print(f"X^T X condition number: {np.linalg.cond(XtX):.2e}")
    
    # Compute X^T y
    Xty = X.T @ y
    print(f"X^T y shape: {Xty.shape}")
    
    # Solve: X^T X w = X^T y
    weights = np.linalg.solve(XtX, Xty)
    print(f"\n✅ Solved analytically!")
    print(f"Weights shape: {weights.shape}")
    
    return weights.flatten()


def main():
    print("="*80)
    print("Experiment 016 V3: Analytical PID Recovery")
    print("="*80)
    print("Using least squares: weights = (X^T X)^{-1} X^T y")
    print("="*80)
    
    # Load data using proper split
    data_split = get_data_split()
    train_files = data_split['train']
    
    print(f"\n✓ Using {len(train_files):,} training files")
    
    # Collect demonstrations
    print("\nCollecting PID demonstrations...")
    states, actions = collect_pid_demonstrations_stateful(train_files, num_files=1000)
    
    # Use analytical solution
    weights = analytical_least_squares(states, actions)
    
    # Compare to PID
    print(f"\n{'='*80}")
    print(f"Results: Analytical Least Squares")
    print(f"{'='*80}")
    print(f"\n📊 Learned vs PID coefficients:")
    print(f"   P: {weights[0]:+.6f}  (PID: +0.195000) | Error: {abs(weights[0] - 0.195):.9f}")
    print(f"   I: {weights[1]:+.6f}  (PID: +0.100000) | Error: {abs(weights[1] - 0.100):.9f}")
    print(f"   D: {weights[2]:+.6f}  (PID: -0.053000) | Error: {abs(weights[2] + 0.053):.9f}")
    
    total_error = (abs(weights[0] - 0.195) + abs(weights[1] - 0.100) + abs(weights[2] + 0.053))
    print(f"\n   Total error: {total_error:.9f}")
    
    if total_error < 1e-6:
        print(f"   ✅ PERFECT! Recovered PID exactly (within numerical precision)!")
    elif total_error < 1e-3:
        print(f"   ✓ Very close to PID")
    else:
        print(f"   ⚠️  Significant deviation from PID")
    
    # Save model
    model = OneNeuronNetwork()
    model.linear.weight.data = torch.FloatTensor(weights).unsqueeze(0)
    
    save_path = Path(__file__).parent / 'one_neuron_analytical.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    
    print(f"\n✅ Saved model: {save_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()



