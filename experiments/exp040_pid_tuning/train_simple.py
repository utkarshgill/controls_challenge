"""
Experiment 040: SIMPLE BC - exact copy of exp016 approach

Just BC. THAT'S IT.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator


class OneNeuronNetwork(nn.Module):
    """1 neuron: 3 weights, no bias"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)


def collect_pid_demonstrations(data_files, num_files=1000):
    """Collect state-action pairs - EXACT copy from exp016"""
    # Check cache
    cache_path = Path(__file__).parent / f'pid_demonstrations_{num_files}.pkl'
    if cache_path.exists():
        print(f"\n✓ Loading cached demonstrations from {cache_path.name}")
        import pickle
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            return data['states'], data['actions']
    
    print(f"\n{'='*80}")
    print(f"Collecting PID Demonstrations")
    print(f"{'='*80}")
    
    MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    model = TinyPhysicsModel(str(MODEL_PATH), debug=False)
    
    states = []
    actions = []
    
    for data_file in tqdm(data_files[:num_files], desc="Collecting"):
        controller = PIDController()
        sim = TinyPhysicsSimulator(model, str(data_file), controller=controller)
        
        # Monkey patch to capture the EXACT state PID uses
        original_update = controller.update
        def capture_update(target_lataccel, current_lataccel, state, future_plan):
            # Save state BEFORE PID modifies it
            old_error_integral = controller.error_integral
            old_prev_error = controller.prev_error
            
            # Compute what PID will compute
            error = target_lataccel - current_lataccel
            
            # Reconstruct exactly what PID will use:
            new_error_integral = old_error_integral + error
            error_diff = error - old_prev_error
            
            # State that PID uses: [error, NEW error_integral, error_diff]
            state_vec = [error, new_error_integral, error_diff]
            
            # Call PID to get action
            action = original_update(target_lataccel, current_lataccel, state, future_plan)
            
            # Verify (sanity check)
            expected_action = (controller.p * error + 
                             controller.i * new_error_integral + 
                             controller.d * error_diff)
            if abs(action - expected_action) > 1e-5:
                print(f"WARNING: Mismatch! action={action:.6f}, expected={expected_action:.6f}")
            
            states.append(state_vec)
            actions.append(action)
            
            return action
        
        controller.update = capture_update
        sim.rollout()
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
    
    print(f"\n✓ Collected {len(states):,} state-action pairs")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Save cache
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump({'states': states, 'actions': actions}, f)
    print(f"✓ Saved cache to {cache_path.name}")
    
    return states, actions


def train_one_neuron_analytical(states, actions):
    """Use analytical least squares - NO optimization, EXACT solution"""
    print(f"\n{'='*80}")
    print(f"Analytical Least Squares Solution")
    print(f"{'='*80}")
    print(f"Expected weights: P=0.195, I=0.100, D=-0.053")
    print(f"{'='*80}\n")
    
    # Analytical solution: weights = (X^T X)^{-1} X^T y
    print(f"X shape: {states.shape}")
    print(f"y shape: {actions.shape}")
    
    XtX = states.T @ states
    Xty = states.T @ actions
    weights = np.linalg.solve(XtX, Xty).flatten()
    
    print(f"✅ Solved analytically!")
    
    # Print results
    print(f"\n📊 Learned vs PID:")
    print(f"   P: {weights[0]:+.6f}  (PID: +0.195000) | Δ={abs(weights[0] - 0.195):.9f}")
    print(f"   I: {weights[1]:+.6f}  (PID: +0.100000) | Δ={abs(weights[1] - 0.100):.9f}")
    print(f"   D: {weights[2]:+.6f}  (PID: -0.053000) | Δ={abs(weights[2] + 0.053):.9f}")
    
    total_error = (abs(weights[0] - 0.195) + abs(weights[1] - 0.100) + abs(weights[2] + 0.053))
    print(f"\n   Total Δ: {total_error:.9f}")
    
    if total_error < 0.001:
        print(f"   ✅ PERFECT! BC successfully cloned PID!")
    elif total_error < 0.01:
        print(f"   ✓ Very close")
    else:
        print(f"   ❌ FAILED")
    
    # Create model with these weights
    model = OneNeuronNetwork()
    model.linear.weight.data = torch.FloatTensor(weights).unsqueeze(0)
    
    return model


def main():
    print("="*80)
    print("Exp040 SIMPLE: Just BC with SGD")
    print("="*80)
    
    # Load data
    DATA_PATH = Path(__file__).parent.parent.parent / 'data'
    all_files = sorted(DATA_PATH.glob('*.csv'))
    
    import random
    random.seed(42)
    random.shuffle(all_files)
    train_files = all_files[:15000]
    
    # Collect
    states, actions = collect_pid_demonstrations(train_files, num_files=1000)
    
    # Train ANALYTICALLY
    model = train_one_neuron_analytical(states, actions)
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

