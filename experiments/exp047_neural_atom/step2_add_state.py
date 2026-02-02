"""
Experiment 47: Neural Atom
Step 2: Add state features to improve beyond PID

Input: [error, error_integral, error_diff, roll_lataccel, v_ego, a_ego]
Output: action = w · features

Can the neuron discover that roll/v_ego/a_ego contain useful information?
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController, STEER_RANGE


class PIDDataCollector(BaseController):
    """Collect (features, action) from PID"""
    
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        self.features = []
        self.actions = []
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Store with state
        self.features.append([
            error,
            self.error_integral,
            error_diff,
            state.roll_lataccel,
            state.v_ego / 34.0,  # Normalize
            state.a_ego
        ])
        self.actions.append(action)
        
        return action


class LearnedController(BaseController):
    """Use learned weights"""
    
    def __init__(self, weights):
        self.w = weights
        self.error_integral = 0
        self.prev_error = 0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        features = np.array([
            error,
            self.error_integral,
            error_diff,
            state.roll_lataccel,
            state.v_ego / 34.0,
            state.a_ego
        ])
        
        action = self.w @ features
        return float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))


if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("STEP 2: Add state features")
    print("="*60)
    
    # Collect data with state
    print("\nCollecting PID data with state features...")
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = PIDDataCollector()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost_pid = sim.rollout()
    
    print(f"  PID cost: {cost_pid['total_cost']:.2f}")
    print(f"  Samples: {len(controller.features)}")
    
    # Train with state
    X = np.array(controller.features)
    y = np.array(controller.actions)
    
    print(f"\nTraining neuron with 6 features...")
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    
    print(f"\n{'='*60}")
    print(f"LEARNED WEIGHTS:")
    print(f"{'='*60}")
    print(f"  error:          {w[0]:.6f}  (PID: 0.195)")
    print(f"  error_integral: {w[1]:.6f}  (PID: 0.100)")
    print(f"  error_diff:     {w[2]:.6f}  (PID: -0.053)")
    print(f"  roll_lataccel:  {w[3]:.6f}  (NEW)")
    print(f"  v_ego:          {w[4]:.6f}  (NEW)")
    print(f"  a_ego:          {w[5]:.6f}  (NEW)")
    
    # Check fit
    y_pred = X @ w
    r2 = 1 - np.mean((y - y_pred)**2) / np.var(y)
    print(f"\nFit quality: R² = {r2:.9f}")
    
    # Test if learned controller matches PID
    print(f"\n{'='*60}")
    print(f"TESTING LEARNED CONTROLLER:")
    print(f"{'='*60}")
    
    learned_ctrl = LearnedController(w)
    sim2 = TinyPhysicsSimulator(model, str(data_path), learned_ctrl, debug=False)
    cost_learned = sim2.rollout()
    
    print(f"\nPID cost:     {cost_pid['total_cost']:.2f}")
    print(f"Learned cost: {cost_learned['total_cost']:.2f}")
    print(f"Difference:   {abs(cost_learned['total_cost'] - cost_pid['total_cost']):.2f}")
    
    if abs(cost_learned['total_cost'] - cost_pid['total_cost']) < 0.1:
        print(f"\n✓ Learned controller EXACTLY matches PID!")
        print(f"\nConclusion: State features (roll, v_ego, a_ego) don't help.")
        print(f"            PID already optimal for this problem structure.")
    else:
        print(f"\n⚠ Learned controller differs from PID")
        print(f"   This suggests state features ARE useful!")
    
    # Save
    weights_file = Path(__file__).parent / 'pid_plus_state_weights.npy'
    np.save(weights_file, w)
    print(f"\nSaved weights to: {weights_file}")
