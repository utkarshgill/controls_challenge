"""
Behavioral Cloning for Controls Challenge
Trains MLP to imitate PID controller with future plan curvatures.

Architecture: 55 -> 64 -> 32 -> 1 (with tanh activations)
Input features:
  - 5 base: [error, error_integral, error_diff, current_curvature, target_curvature]
  - 50 future curvatures: [future_curvature_t+1, ..., future_curvature_t+50]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import random
import types
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for behavioral cloning.
    Architecture: 55 -> 64 -> 32 -> 1 with tanh activations
    """
    def __init__(self, input_dim=55, hidden_sizes=[64, 32]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0], bias=True)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=True)
        self.fc3 = nn.Linear(hidden_sizes[1], 1, bias=True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x.squeeze(-1)


def collect_data_from_files(file_list, model, desc="Collecting"):
    """
    Collect training data from PID controller rollouts.
    
    Returns:
        X: array of shape (N, 55) with [5 base features + 50 future curvatures]
        Y: array of shape (N,) with PID outputs (steer commands)
    """
    all_X = []
    all_Y = []
    
    for file_path in tqdm(file_list, desc=desc):
        # Create fresh PID controller
        data_controller = pid.Controller()
        
        # Lists to collect data for this rollout
        Xs_rollout = []
        Ys_rollout = []
        
        # Define data collection function
        def update_with_data_collection(self, target_lataccel, current_lataccel, state, future_plan):
            error = (target_lataccel - current_lataccel)
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            
            # Compute current curvatures: (lat - roll) / v^2
            v_ego_squared = state.v_ego ** 2
            if v_ego_squared > 0.01:
                current_curvature = (current_lataccel - state.roll_lataccel) / v_ego_squared
                target_curvature = (target_lataccel - state.roll_lataccel) / v_ego_squared
            else:
                current_curvature = 0.0
                target_curvature = 0.0
            
            # Compute future plan curvatures
            future_curvatures = []
            for i in range(len(future_plan.lataccel)):
                v_future_squared = future_plan.v_ego[i] ** 2
                if v_future_squared > 0.01:
                    future_curv = (future_plan.lataccel[i] - future_plan.roll_lataccel[i]) / v_future_squared
                else:
                    future_curv = 0.0
                future_curvatures.append(future_curv)
            
            # Pad to 50 features
            while len(future_curvatures) < 50:
                future_curvatures.append(0.0)
            future_curvatures = future_curvatures[:50]
            
            # Store inputs: base features + future curvatures
            features = [error, self.error_integral, error_diff, current_curvature, target_curvature] + future_curvatures
            Xs_rollout.append(features)
            
            # Compute PID output
            out = self.p * error + self.i * self.error_integral + self.d * error_diff
            Ys_rollout.append(out)
            
            return out
        
        # Bind the data collection method
        data_controller.update = types.MethodType(update_with_data_collection, data_controller)
        
        # Run simulator to collect data
        sim = TinyPhysicsSimulator(model, str(file_path), controller=data_controller, debug=False)
        sim.rollout()
        
        # Accumulate data
        all_X.extend(Xs_rollout)
        all_Y.extend(Ys_rollout)
    
    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32)


def train_bc(model_path, data_dir, output_dir, 
             num_train_files=800, num_val_files=100, num_test_files=100,
             num_epochs=50, batch_size=256, lr=0.001, seed=42):
    """
    Train behavioral cloning model.
    """
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get data files
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    random.shuffle(all_files)
    
    total_files = num_train_files + num_val_files + num_test_files
    selected_files = all_files[:total_files]
    
    train_files = selected_files[:num_train_files]
    val_files = selected_files[num_train_files:num_train_files + num_val_files]
    test_files = selected_files[num_train_files + num_val_files:]
    
    print(f"Data split:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")
    
    # Load TinyPhysics model
    print(f"\nLoading TinyPhysics model from {model_path}...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Collect data
    print(f"\nCollecting training data...")
    X_train, Y_train = collect_data_from_files(train_files, tinyphysics_model, desc="Train")
    
    print(f"\nCollecting validation data...")
    X_val, Y_val = collect_data_from_files(val_files, tinyphysics_model, desc="Val")
    
    print(f"\nCollecting test data...")
    X_test, Y_test = collect_data_from_files(test_files, tinyphysics_model, desc="Test")
    
    print(f"\nData collection complete!")
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)
    X_test_t = torch.FloatTensor(X_test)
    Y_test_t = torch.FloatTensor(Y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    mlp = MLP(input_dim=55, hidden_sizes=[64, 32])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    
    print(f"\nModel architecture:")
    print(mlp)
    print(f"Total parameters: {sum(p.numel() for p in mlp.parameters())}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training phase
        mlp.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = mlp(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = mlp(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(mlp.state_dict(), output_path / 'best_model.pt')
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Test on held-out set
    mlp.eval()
    with torch.no_grad():
        test_preds = mlp(X_test_t)
        test_loss = criterion(test_preds, Y_test_t).item()
    
    print(f"Test Loss (MSE): {test_loss:.6f}")
    
    # Save final model and training history
    torch.save(mlp.state_dict(), output_path / 'final_model.pt')
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'best_epoch': best_epoch,
        'config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'architecture': '55->64->32->1'
        }
    }, output_path / 'training_history.pt')
    
    print(f"\nModel saved to {output_path}")
    print(f"  - best_model.pt (epoch {best_epoch})")
    print(f"  - final_model.pt")
    print(f"  - training_history.pt")
    
    return mlp


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BC model for controls challenge')
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx',
                        help='Path to TinyPhysics model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing CSV data files')
    parser.add_argument('--output_dir', type=str, default='./experiments/exp042/outputs',
                        help='Directory to save trained models')
    parser.add_argument('--num_train', type=int, default=800,
                        help='Number of training files')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Number of validation files')
    parser.add_argument('--num_test', type=int, default=100,
                        help='Number of test files')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    train_bc(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_files=args.num_train,
        num_val_files=args.num_val,
        num_test_files=args.num_test,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
