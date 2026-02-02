"""
Behavioral Cloning with 1D Convolutions for future trajectory.

Architecture:
  - Base features (5): MLP path
  - Future curvatures (50): Conv1D path  
  - Combine and predict
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

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid


class MLPConv(nn.Module):
    """
    MLP with 1D Conv for temporal future trajectory.
    
    Architecture:
      - Base path (5 features): 5 -> 32
      - Future path (50 curvatures): Conv1D -> 32
      - Combined: 64 -> 32 -> 1
    """
    def __init__(self):
        super().__init__()
        
        # Base features path (5 -> 32)
        self.base_fc1 = nn.Linear(5, 16, bias=True)
        self.base_fc2 = nn.Linear(16, 32, bias=True)
        
        # Future curvatures path (50 -> 32)
        # Input: (batch, 50) -> (batch, 1, 50) for Conv1D
        self.future_conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)  # -> (batch, 16, 50)
        self.future_conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)   # -> (batch, 8, 50)
        self.future_pool = nn.AdaptiveAvgPool1d(4)  # -> (batch, 8, 4) = 32 features
        self.future_fc = nn.Linear(32, 32, bias=True)
        
        # Combined path (64 -> 32 -> 1)
        self.combine_fc1 = nn.Linear(64, 32, bias=True)
        self.output = nn.Linear(32, 1, bias=True)
        
    def forward(self, x):
        # Split input
        base = x[:, :5]          # (batch, 5)
        future = x[:, 5:]        # (batch, 50)
        
        # Base path
        base_out = torch.tanh(self.base_fc1(base))
        base_out = torch.tanh(self.base_fc2(base_out))  # (batch, 32)
        
        # Future path - add channel dimension for Conv1D
        future = future.unsqueeze(1)  # (batch, 1, 50)
        future_out = torch.tanh(self.future_conv1(future))  # (batch, 16, 50)
        future_out = torch.tanh(self.future_conv2(future_out))  # (batch, 8, 50)
        future_out = self.future_pool(future_out)  # (batch, 8, 4)
        future_out = future_out.view(future_out.size(0), -1)  # (batch, 32)
        future_out = torch.tanh(self.future_fc(future_out))  # (batch, 32)
        
        # Combine paths
        combined = torch.cat([base_out, future_out], dim=1)  # (batch, 64)
        combined = torch.tanh(self.combine_fc1(combined))  # (batch, 32)
        output = torch.tanh(self.output(combined))  # (batch, 1)
        
        return output.squeeze(-1)


def collect_data_from_files(file_list, model, desc="Collecting"):
    """Same as train_bc.py"""
    all_X = []
    all_Y = []
    
    for file_path in tqdm(file_list, desc=desc):
        data_controller = pid.Controller()
        Xs_rollout = []
        Ys_rollout = []
        
        def update_with_data_collection(self, target_lataccel, current_lataccel, state, future_plan):
            error = (target_lataccel - current_lataccel)
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            
            v_ego_squared = state.v_ego ** 2
            if v_ego_squared > 0.01:
                current_curvature = (current_lataccel - state.roll_lataccel) / v_ego_squared
                target_curvature = (target_lataccel - state.roll_lataccel) / v_ego_squared
            else:
                current_curvature = 0.0
                target_curvature = 0.0
            
            future_curvatures = []
            for i in range(len(future_plan.lataccel)):
                v_future_squared = future_plan.v_ego[i] ** 2
                if v_future_squared > 0.01:
                    future_curv = (future_plan.lataccel[i] - future_plan.roll_lataccel[i]) / v_future_squared
                else:
                    future_curv = 0.0
                future_curvatures.append(future_curv)
            
            while len(future_curvatures) < 50:
                future_curvatures.append(0.0)
            future_curvatures = future_curvatures[:50]
            
            features = [error, self.error_integral, error_diff, current_curvature, target_curvature] + future_curvatures
            Xs_rollout.append(features)
            
            out = self.p * error + self.i * self.error_integral + self.d * error_diff
            Ys_rollout.append(out)
            
            return out
        
        data_controller.update = types.MethodType(update_with_data_collection, data_controller)
        sim = TinyPhysicsSimulator(model, str(file_path), controller=data_controller, debug=False)
        sim.rollout()
        
        all_X.extend(Xs_rollout)
        all_Y.extend(Ys_rollout)
    
    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32)


def train_bc_conv(model_path, data_dir, output_dir, 
                  num_train_files=800, num_val_files=100, num_test_files=100,
                  num_epochs=50, batch_size=256, lr=0.001, seed=42):
    """Train BC with Conv1D architecture"""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
    
    # Initialize Conv model
    mlp = MLPConv()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    
    print(f"\nModel architecture (with Conv1D):")
    print(mlp)
    print(f"Total parameters: {sum(p.numel() for p in mlp.parameters())}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training
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
        
        # Validation
        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = mlp(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(mlp.state_dict(), output_path / 'best_model_conv.pt')
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Test
    mlp.eval()
    with torch.no_grad():
        test_preds = mlp(X_test_t)
        test_loss = criterion(test_preds, Y_test_t).item()
    
    print(f"Test Loss (MSE): {test_loss:.6f}")
    
    # Save
    torch.save(mlp.state_dict(), output_path / 'final_model_conv.pt')
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'best_epoch': best_epoch,
        'architecture': 'MLPConv (base MLP + future Conv1D)'
    }, output_path / 'training_history_conv.pt')
    
    print(f"\nModel saved to {output_path}")
    print(f"  - best_model_conv.pt")
    print(f"  - final_model_conv.pt")
    
    return mlp


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BC with Conv1D')
    parser.add_argument('--model_path', type=str, default='../../models/tinyphysics.onnx')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_train', type=int, default=800)
    parser.add_argument('--num_val', type=int, default=100)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_bc_conv(
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
