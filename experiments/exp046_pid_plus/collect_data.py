"""
Collect training data for neural controller
Run on multiple trajectories, record features and actions
"""

import numpy as np
import sys
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController


class DataCollector(BaseController):
    """v3 controller that logs data"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.prev_action = 0.0
        
        # Feedforward params (from v3)
        self.ff_immediate = 0.3
        self.ff_derivative = 0.15
        
        # Data storage
        self.data = []
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Standard PID
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Feedforward
        ff_action = 0.0
        
        if len(future_plan.lataccel) >= 5:
            # Features
            immediate_error = future_plan.lataccel[0] - target_lataccel
            future_slice = future_plan.lataccel[:5]
            target_derivative = (future_slice[-1] - future_slice[0]) / 5.0
            
            # Record features and target
            features = np.array([
                immediate_error,      # Feature 0: next step error
                target_derivative,    # Feature 1: rate of change
                error,                # Feature 2: current error
                self.prev_action,     # Feature 3: previous action (smoothness)
                state.v_ego / 34.0,   # Feature 4: velocity (normalized)
                state.roll_lataccel   # Feature 5: road roll
            ])
            
            # Compute feedforward (what v3 does)
            ff_action = self.ff_immediate * immediate_error + self.ff_derivative * target_derivative
            
            # Store (features, ff_action)
            self.data.append({
                'features': features,
                'ff_action': ff_action,  # This is what we want to learn
                'pid_action': pid_action
            })
        
        action = pid_action + ff_action
        self.prev_action = action
        return action


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent.parent / 'data'
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    
    # Collect from first 20 trajectories
    all_data = []
    
    print("Collecting training data...")
    print("="*60)
    
    for i in range(20):
        csv_file = data_dir / f'{i:05d}.csv'
        if not csv_file.exists():
            break
            
        print(f"  Route {i:05d}...", end='', flush=True)
        
        model = TinyPhysicsModel(str(model_path), debug=False)
        controller = DataCollector()
        sim = TinyPhysicsSimulator(model, str(csv_file), controller=controller, debug=False)
        cost = sim.rollout()
        
        all_data.extend(controller.data)
        print(f" {len(controller.data)} samples, cost={cost['total_cost']:.1f}")
    
    print(f"\n{'='*60}")
    print(f"Collected {len(all_data)} training samples")
    
    # Save
    output_file = Path(__file__).parent / 'training_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"Saved to: {output_file}")
    
    # Show statistics
    features = np.array([d['features'] for d in all_data])
    ff_actions = np.array([d['ff_action'] for d in all_data])
    
    print(f"\nFeature statistics:")
    feature_names = ['immediate_err', 'derivative', 'current_err', 'prev_action', 'v_ego', 'roll']
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: mean={features[:,i].mean():7.4f}, std={features[:,i].std():7.4f}")
    
    print(f"\nFF Action: mean={ff_actions.mean():.4f}, std={ff_actions.std():.4f}")
