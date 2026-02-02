"""
Step 2: PID + 1-Step Lookahead

Add simplest possible MPC:
- Look 1 step ahead
- Try 10 random actions
- Pick best one

This is PID with minimal search.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, BaseController, 
                         STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER, DEL_T, 
                         CONTEXT_LENGTH, MAX_ACC_DELTA, State)


class MPC1Step(BaseController):
    """PID + 1-step lookahead search"""
    
    def __init__(self, model, num_samples=20):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        
        # PID fallback for first 20 steps
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # History for model
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        
    def update(self, target, current, state, future_plan):
        # Fallback to PID if no history
        if len(self.action_history) < CONTEXT_LENGTH or len(future_plan.lataccel) == 0:
            error = target - current
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            action = self.p * error + self.i * self.error_integral + self.d * error_diff
            action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
            self.state_history.append(state)
            self.pred_history.append(current)
            self.action_history.append(action)
            return action
        
        # Get PID baseline action
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        pid_action = np.clip(pid_action, STEER_RANGE[0], STEER_RANGE[1])
        
        # Sample around PID (small perturbations)
        actions = pid_action + np.random.randn(self.num_samples) * 0.1
        actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
        
        # Evaluate each
        costs = []
        prev_action = self.action_history[-1] if len(self.action_history) > 0 else 0.0
        
        for action in actions:
            # Simulate 1 step forward
            sim_actions = list(self.action_history[-CONTEXT_LENGTH:]) + [action]
            next_pred = self.model.get_current_lataccel(
                sim_states=self.state_history[-CONTEXT_LENGTH:],
                actions=sim_actions[-CONTEXT_LENGTH:],
                past_preds=self.pred_history[-CONTEXT_LENGTH:]
            )
            next_pred = np.clip(next_pred, current - MAX_ACC_DELTA, current + MAX_ACC_DELTA)
            
            # Cost = tracking error + jerk + action smoothness
            target_next = future_plan.lataccel[0]
            lat_cost = ((next_pred - target_next) ** 2) * LAT_ACCEL_COST_MULTIPLIER
            jerk = (next_pred - current) / DEL_T
            jerk_cost = jerk ** 2
            smooth_cost = ((action - prev_action) ** 2) * 10  # Penalize jumps
            
            cost = lat_cost + jerk_cost + smooth_cost
            costs.append(cost)
        
        # Pick best
        best_idx = np.argmin(costs)
        best_action = float(actions[best_idx])
        
        self.state_history.append(state)
        self.pred_history.append(current)
        self.action_history.append(best_action)
        return best_action


if __name__ == '__main__':
    import time
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("STEP 2: PID + 1-Step Lookahead (20 samples)")
    print("="*60)
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = MPC1Step(model, num_samples=20)
    
    print("\nRunning...")
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Cost: {cost['total_cost']:.2f}")
    print(f"    Lat: {cost['lataccel_cost']:.2f} × 50 = {cost['lataccel_cost']*50:.1f}")
    print(f"    Jerk: {cost['jerk_cost']:.2f}")
    print(f"  Time: {(t1-t0):.1f}s")
    print(f"{'='*60}")
    print(f"\nPID: 102.9")
    print(f"This: {cost['total_cost']:.1f}")
    
    if cost['total_cost'] < 102:
        print("✓ Better than PID!")
    else:
        gap = cost['total_cost'] - 102.9
        print(f"✗ Worse by {gap:.1f}")
