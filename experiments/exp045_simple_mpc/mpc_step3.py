"""
Step 3: Horizon=10, Simple Random Shooting

Increase horizon to 10 steps.
Keep it simple: just random shooting (no CEM yet).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, BaseController, 
                         STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER, DEL_T, 
                         CONTEXT_LENGTH, MAX_ACC_DELTA, State)


class SimpleMPC(BaseController):
    
    def __init__(self, model, horizon=10, samples=50):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.samples = samples
        
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        
        self.timestep = 0
        
    def update(self, target, current, state, future_plan):
        self.timestep += 1
        if self.timestep % 100 == 0:
            print(f"  Step {self.timestep}...", flush=True)
        
        H = min(self.horizon, len(future_plan.lataccel))
        
        # Fallback
        if H == 0 or len(self.action_history) < CONTEXT_LENGTH:
            action = 0.195 * (target - current)
            action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
            self.state_history.append(state)
            self.pred_history.append(current)
            self.action_history.append(action)
            return action
        
        # Sample action sequences
        seqs = np.random.randn(self.samples, H) * 0.3
        seqs = np.clip(seqs, STEER_RANGE[0], STEER_RANGE[1])
        
        # Evaluate
        best_cost = float('inf')
        best_seq = seqs[0]  # Initialize with first sequence
        
        for seq in seqs:
            cost = self._eval(seq, future_plan, current, H)
            if np.isfinite(cost) and cost < best_cost:
                best_cost = cost
                best_seq = seq
        
        action = float(best_seq[0])
        self.state_history.append(state)
        self.pred_history.append(current)
        self.action_history.append(action)
        return action
    
    def _eval(self, actions, future_plan, current, H):
        # Copy context
        states = list(self.state_history[-CONTEXT_LENGTH:])
        sim_actions = list(self.action_history[-CONTEXT_LENGTH:])
        preds = list(self.pred_history[-CONTEXT_LENGTH:])
        
        pred_lat = []
        curr = current
        
        for t in range(H):
            next_pred = self.model.get_current_lataccel(
                sim_states=states[-CONTEXT_LENGTH:],
                actions=sim_actions[-CONTEXT_LENGTH:],
                past_preds=preds[-CONTEXT_LENGTH:]
            )
            next_pred = np.clip(next_pred, curr - MAX_ACC_DELTA, curr + MAX_ACC_DELTA)
            pred_lat.append(next_pred)
            
            sim_actions.append(actions[t])
            preds.append(next_pred)
            states.append(State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t]
            ))
            curr = next_pred
        
        # Cost
        target = np.array(future_plan.lataccel[:H])
        pred = np.array(pred_lat)
        
        lat_cost = np.mean((target - pred) ** 2) * 100 * LAT_ACCEL_COST_MULTIPLIER
        jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
        
        return lat_cost + jerk_cost


if __name__ == '__main__':
    import time
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("STEP 3: Horizon=10, Random Shooting (50 samples)")
    print("="*60)
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = SimpleMPC(model, horizon=10, samples=50)
    
    print("\nRunning (should take ~3 min)...")
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Cost: {cost['total_cost']:.2f}")
    print(f"    Lat: {cost['lataccel_cost']:.2f} × 50 = {cost['lataccel_cost']*50:.1f}")
    print(f"    Jerk: {cost['jerk_cost']:.2f}")
    print(f"  Time: {(t1-t0)/60:.1f} min")
    print(f"{'='*60}")
    print(f"\nPID: 102.9")
    print(f"This: {cost['total_cost']:.1f}")
    
    if cost['total_cost'] < 102:
        print("✓ Better than PID!")
    else:
        gap = cost['total_cost'] - 102.9
        print(f"✗ Gap: {gap:.1f}")
