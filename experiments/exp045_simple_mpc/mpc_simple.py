"""
MPC with CEM - SIMPLE VERSION THAT WORKS

Key simplifications:
- Exact cost from tinyphysics.py (no smoothness)
- Reduced compute for speed
- No fancy batching
"""

import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import (TinyPhysicsModel, BaseController, LAT_ACCEL_COST_MULTIPLIER, 
                         DEL_T, CONTEXT_LENGTH, MAX_ACC_DELTA, STEER_RANGE, State)

NUM_WORKERS = max(1, int(os.cpu_count() * 0.75))


def compute_cost(target, pred):
    """EXACT cost from tinyphysics.py"""
    lat_cost = np.mean((target - pred) ** 2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
    return (lat_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost


def rollout(model, actions, states, sim_actions, sim_preds, current_lat, future_plan, H):
    """Rollout actions through model"""
    states = list(states)
    sim_actions = list(sim_actions)
    sim_preds = list(sim_preds)
    
    pred = np.zeros(H)
    current = current_lat
    
    for t in range(H):
        next_pred = model.get_current_lataccel(
            sim_states=states[-CONTEXT_LENGTH:],
            actions=sim_actions[-CONTEXT_LENGTH:],
            past_preds=sim_preds[-CONTEXT_LENGTH:]
        )
        next_pred = np.clip(next_pred, current - MAX_ACC_DELTA, current + MAX_ACC_DELTA)
        pred[t] = next_pred
        
        # Clip actions to match simulator behavior
        action = np.clip(actions[t], STEER_RANGE[0], STEER_RANGE[1])
        sim_actions.append(action)
        sim_preds.append(next_pred)
        states.append(State(
            roll_lataccel=future_plan.roll_lataccel[t],
            v_ego=future_plan.v_ego[t],
            a_ego=future_plan.a_ego[t]
        ))
        current = next_pred
    
    return pred


class MPC(BaseController):
    def __init__(self, model, H_control=12, H_predict=50, samples=150, elites=30, iters=4, std=0.12, seed=0):
        super().__init__()
        self.model = model
        self.H_control = H_control
        self.H_predict = H_predict
        self.samples = samples
        self.elites = elites
        self.iters = iters
        self.std = std
        self.seed = seed
        
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        self.prev_mean = None
        self.timestep = 0

        # Running cost tracking (matches tinyphysics.py)
        self.running_lat_cost = 0.0
        self.running_jerk_cost = 0.0
        self.running_steps = 0
        self.prev_lat = None

        # Deterministic CEM for stability
        np.random.seed(self.seed)
        
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        
    def update(self, target, current, state, future_plan):
        self.timestep += 1
        # Running cost (starts at step 100 per challenge rules)
        if self.timestep >= 100:
            lat_error = (target - current) ** 2
            self.running_lat_cost += lat_error
            jerk_val = 0.0
            if self.prev_lat is not None:
                jerk_val = (current - self.prev_lat) / DEL_T
            self.running_jerk_cost += jerk_val ** 2
            self.running_steps += 1

            if self.timestep % 50 == 0:
                avg_lat = (self.running_lat_cost / self.running_steps) * 100 * LAT_ACCEL_COST_MULTIPLIER
                avg_jerk = (self.running_jerk_cost / self.running_steps) * 100
                total = avg_lat + avg_jerk
                print(f"  Step {self.timestep}: cost={total:.1f} (lat={avg_lat:.1f}, jerk={avg_jerk:.1f})", flush=True)
        elif self.timestep % 50 == 0:
            print(f"  Step {self.timestep}...", flush=True)

        self.prev_lat = current
        
        H_pred = min(self.H_predict, len(future_plan.lataccel))
        H_ctrl = min(self.H_control, H_pred)
        
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
            
            # Extend to H_pred
            if H_pred > H_ctrl:
                seqs_ext = np.column_stack([seqs, np.repeat(seqs[:, -1:], H_pred - H_ctrl, axis=1)])
            else:
                seqs_ext = seqs
            seqs_ext = np.clip(seqs_ext, STEER_RANGE[0], STEER_RANGE[1])
            
            # Evaluate
            costs = np.array(list(self.executor.map(
                lambda seq: self._eval(seq, future_plan, current, H_pred),
                seqs_ext
            )))
            
            # Update
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
        pred = rollout(
            self.model, actions,
            self.state_history[-CONTEXT_LENGTH:],
            self.action_history[-CONTEXT_LENGTH:],
            self.pred_history[-CONTEXT_LENGTH:],
            current, future_plan, H
        )
        target = np.array(future_plan.lataccel[:H])
        return compute_cost(target, pred)


if __name__ == '__main__':
    from tinyphysics import TinyPhysicsSimulator
    import time
    
    print("="*60)
    print("MPC - SIMPLE & RELIABLE (<90 target)")
    print("="*60)
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("\nLoading model...")
    model = TinyPhysicsModel(str(model_path), debug=False)
    
    print("\nConfig:")
    print("  Control: 12 actions")
    print("  Predict: 50 steps (ALL future)")
    print("  Samples: 150, Elites: 30, Iters: 4")
    print("  Std: 0.12 (stable exploration)")
    print("  Seed: 0 (deterministic)")
    print("  Cost: EXACT (lat×50 + jerk)")
    print("  Total: 150×50×4 = 30,000 calls/step")
    print("  Expected: ~20-25 min")
    print()
    
    controller = MPC(model, H_control=12, H_predict=50, samples=150, elites=30, iters=4, std=0.12, seed=0)
    
    print("Running...\n")
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
    
    if cost['total_cost'] < 100:
        print("\n✓ SUCCESS! <100")
    else:
        print(f"\n✗ Gap: {cost['total_cost']-100:.1f}")
