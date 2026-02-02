"""
Experiment 016 V2: Stateful One-Neuron Controller

NOW with correct inputs that match PID:
- Input: [error, error_integral, error_diff]
- Output: steering
- Expected weights: [0.195, 0.100, -0.053] (matching PID!)

This SHOULD recover PID coefficients exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from controllers.pid import Controller as PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator


class OneNeuronNetwork(nn.Module):
    """1 neuron: 3 weights, no bias"""
    def __init__(self):
        super().__init__()
        # Single linear layer with NO bias
        self.linear = nn.Linear(3, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)


def collect_pid_demonstrations_stateful(data_files, num_files=1000):
    """Collect state-action pairs WITH PID's internal state"""
    print(f"\n{'='*80}")
    print(f"Collecting PID Demonstrations (Stateful)")
    print(f"{'='*80}")
    
    model = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)
    
    states = []
    actions = []
    
    for data_file in tqdm(data_files[:num_files], desc="Collecting"):
        controller = PIDController()
        sim = TinyPhysicsSimulator(model, data_file, controller=controller)
        
        # Monkey patch to capture the EXACT state PID uses
        original_update = controller.update
        def capture_update(target_lataccel, current_lataccel, state, future_plan):
            # Save state BEFORE PID modifies it
            old_error_integral = controller.error_integral
            old_prev_error = controller.prev_error
            
            # Compute what PID will compute
            error = target_lataccel - current_lataccel
            
            # Now we can reconstruct exactly what PID will use:
            # error_integral (NEW) = old + error
            new_error_integral = old_error_integral + error
            # error_diff = error - old_prev_error
            error_diff = error - old_prev_error
            
            # State that PID uses: [error, NEW error_integral, error_diff]
            state_vec = [error, new_error_integral, error_diff]
            
            # Call PID to get action (and update its state)
            action = original_update(target_lataccel, current_lataccel, state, future_plan)
            
            # Verify our reconstruction matches (sanity check)
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
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    return states, actions


def train_one_neuron(states, actions, epochs=100, lr=0.01):
    """Train 1-neuron network to recover PID coefficients"""
    print(f"\n{'='*80}")
    print(f"Training One-Neuron Network to Recover PID")
    print(f"{'='*80}")
    print(f"Parameters: 3 weights, 0 bias")
    print(f"Expected weights: P=0.195, I=0.100, D=-0.053")
    print(f"Epochs: {epochs}")
    print(f"LR: {lr}")
    print(f"{'='*80}\n")
    
    # NO NORMALIZATION - we want to recover exact PID coefficients!
    
    # Convert to tensors
    X = torch.FloatTensor(states)
    y = torch.FloatTensor(actions)
    
    # Create model
    model = OneNeuronNetwork()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Use SGD for stability
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 10 == 0:
            weights = model.linear.weight.data.numpy()[0]
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.8f} | "
                  f"P={weights[0]:+.6f} I={weights[1]:+.6f} D={weights[2]:+.6f}")
    
    print(f"\n✅ Training complete!")
    print(f"   Final loss: {loss.item():.8f}")
    print(f"   Best loss: {best_loss:.8f}")
    
    # Print learned weights vs PID
    weights = model.linear.weight.data.numpy()[0]
    print(f"\n📊 Learned vs PID coefficients:")
    print(f"   P: {weights[0]:+.6f}  (PID: +0.195000) | Error: {abs(weights[0] - 0.195):.6f}")
    print(f"   I: {weights[1]:+.6f}  (PID: +0.100000) | Error: {abs(weights[1] - 0.100):.6f}")
    print(f"   D: {weights[2]:+.6f}  (PID: -0.053000) | Error: {abs(weights[2] + 0.053):.6f}")
    
    total_error = (abs(weights[0] - 0.195) + abs(weights[1] - 0.100) + abs(weights[2] + 0.053))
    print(f"\n   Total error: {total_error:.6f}")
    
    if total_error < 0.001:
        print(f"   ✅ PERFECT! Recovered PID exactly!")
    elif total_error < 0.01:
        print(f"   ✓ Very close to PID")
    else:
        print(f"   ⚠️  Significant deviation from PID")
    
    return model


def save_model(model, save_path):
    """Save model checkpoint (no normalization needed)"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"\n✅ Saved model: {save_path}")


def main():
    print("="*80)
    print("Experiment 016 V2: Stateful One-Neuron = PID Recovery")
    print("="*80)
    
    # Load data using proper split
    from data_split import get_data_split
    data_split = get_data_split()
    
    train_files = data_split['train']
    val_files = data_split['val']
    test_files = data_split['test']
    
    print(f"\n✓ Data split loaded:")
    print(f"  Train: {len(train_files):,} files")
    print(f"  Val:   {len(val_files):,} files")
    print(f"  Test:  {len(test_files):,} files")
    
    # Collect demonstrations WITH state
    states, actions = collect_pid_demonstrations_stateful(train_files, num_files=1000)
    
    # Train
    model = train_one_neuron(states, actions, epochs=100, lr=0.01)
    
    # Save
    save_path = Path(__file__).parent / 'one_neuron_stateful.pth'
    save_model(model, save_path)
    
    print(f"\n{'='*80}")
    print(f"✅ If weights match PID, we've successfully recovered it!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

