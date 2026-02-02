"""
Final MPC: Lessons learned

1. Use smoothness weight (500-1000) for stability
2. Use full 50-step prediction horizon
3. Use moderate CEM params (not too aggressive)
4. Warm-start properly
5. Exact cost function
"""

import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator, BaseController,
                         STEER_RANGE, LAT_ACCEL_COST_MULTIPLIER, DEL_T,
                         CONTEXT_LENGTH, MAX_ACC_DELTA, State)

NUM_WORKERS = max(1, int(os.cpu_count() * 0.75))


class FinalMPC(BaseController):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # MPC params (balanced for quality + speed)
        self.H_ctrl = 12
        self.H_pred = 50
        self.samples = 150
        self.elites = 30
        self.iters = 4
        self.std = 0.08  # Smaller = smoother
        self.smoothness_weight = 0  # DO NOT MISALIGN with official cost!
        
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        self.prev_mean = None
        self.timestep = 0
        
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        np.random.seed(0)  # Deterministic
        
    def update(self, target, current, state, future_plan):
        self.timestep += 1
        if self.timestep % 100 == 0:
            print(f"  Step {self.timestep}...", flush=True)
        
        H_pred = min(self.H_pred, len(future_plan.lataccel))
        H_ctrl = min(self.H_ctrl, H_pred)
        
        # Fallback
        if H_pred == 0 or len(self.action_history) < CONTEXT_LENGTH:
            action = 0.195 * (target - current)
            action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
            self.state_history.append(state)
            self.pred_history.append(current)
            self.action_history.append(action)
            return action
        
        # Warm-start
        if self.prev_mean is not None and len(self.prev_mean) >= H_ctrl:
            mean = np.concatenate([self.prev_mean[1:H_ctrl], [0.0]])
        else:
            mean = np.zeros(H_ctrl)
        std = np.ones(H_ctrl) * self.std
        
        # CEM
        for _ in range(self.iters):
            seqs = np.random.randn(self.samples, H_ctrl) * std + mean
            seqs = np.clip(seqs, STEER_RANGE[0], STEER_RANGE[1])
            
            # Extend to H_pred (hold last action)
            if H_pred > H_ctrl:
                seqs_ext = np.column_stack([seqs, np.repeat(seqs[:, -1:], H_pred - H_ctrl, axis=1)])
            else:
                seqs_ext = seqs
            
            # Evaluate in parallel
            costs = np.array(list(self.executor.map(
                lambda seq: self._eval(seq, future_plan, current, H_pred),
                seqs_ext
            )))
            
            # Elites
            elite_idx = np.argsort(costs)[:self.elites]
            elites = seqs[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6
        
        action = float(np.clip(mean[0], STEER_RANGE[0], STEER_RANGE[1]))
        self.prev_mean = mean
        self.state_history.append(state)
        self.pred_history.append(current)
        self.action_history.append(action)
        return action
    
    def _eval(self, actions, future_plan, current, H):
        # Rollout
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
        
        # Official cost (lat + jerk)
        target = np.array(future_plan.lataccel[:H])
        pred = np.array(pred_lat)
        lat_cost = np.mean((target - pred) ** 2) * 100 * LAT_ACCEL_COST_MULTIPLIER
        jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
        
        # Smoothness (internal optimization only)
        prev_action = self.action_history[-1] if len(self.action_history) > 0 else 0.0
        smooth_cost = np.mean((np.diff(actions[:min(len(actions), self.H_ctrl)], prepend=prev_action) ** 2)) * self.smoothness_weight
        
        return lat_cost + jerk_cost + smooth_cost


if __name__ == '__main__':
    import time
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("FINAL MPC: Balanced for <90 target")
    print("="*60)
    print("\nParams:")
    print("  Control: 12, Predict: 50")
    print("  Samples: 150, Elites: 30, Iters: 4")
    print("  Smoothness: 0 (NO MISALIGNMENT!)")
    print("  Std: 0.08 (small = implicit smoothness)")
    print("  Seed: 0, Workers: 9")
    print("  Compute: 150×50×4 = 30,000 calls/step")
    print("  Expected: ~20 min\n")
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = FinalMPC(model)
    
    print("Running...")
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
    
    if cost['total_cost'] < 90:
        print("\n🎉 SUCCESS! <90")
    elif cost['total_cost'] < 100:
        print("\n✓ Good! <100")
    else:
        print(f"\n✗ Gap: {cost['total_cost']-90:.1f} from target")
