"""
Exp045: MPC with CEM - MINIMAL CLEAN VERSION

Removes all unnecessary complexity:
- Only CEM (random shooting removed - it's worse)
- Simplified parallel evaluation
- Clean warm-starting
- No redundant diagnostics
"""

import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import (TinyPhysicsModel, BaseController, LAT_ACCEL_COST_MULTIPLIER, 
                         DEL_T, CONTEXT_LENGTH, MAX_ACC_DELTA, State)


def compute_rollout_cost(target_lataccels, pred_lataccels):
    """
    Compute cost using EXACT formula from tinyphysics.py
    This ensures MPC optimizes the SAME objective it's judged on.
    """
    lat_accel_cost = np.mean((target_lataccels - pred_lataccels) ** 2) * 100
    jerk_cost = np.mean((np.diff(pred_lataccels) / DEL_T) ** 2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return total_cost


def rollout_actions_batched(model, action_sequences, init_states, init_actions, init_preds, init_lataccel, future_plan, H):
    """
    Batched rollout: evaluate ALL sequences together, calling ONNX once per timestep.
    Much faster than evaluating sequences individually.
    
    Args:
        action_sequences: [N, H] array of N action sequences
        
    Returns:
        pred_lats: [N, H] array of predicted lataccels for each sequence
    """
    N = len(action_sequences)
    
    # Initialize contexts for all sequences (they start identical)
    sim_states = [list(init_states) for _ in range(N)]
    sim_actions = [list(init_actions) for _ in range(N)]
    sim_preds = [list(init_preds) for _ in range(N)]
    current_preds = np.full(N, init_lataccel)
    
    pred_lats = np.zeros((N, H))
    
    for t in range(H):
        # Prepare batch inputs
        batch_states = [states[-CONTEXT_LENGTH:] for states in sim_states]
        batch_actions = [actions[-CONTEXT_LENGTH:] for actions in sim_actions]
        batch_preds = [preds[-CONTEXT_LENGTH:] for preds in sim_preds]
        
        # Batch ONNX call - THIS IS THE KEY SPEEDUP!
        next_preds = model.get_current_lataccel_batch(batch_states, batch_actions, batch_preds)
        next_preds = np.clip(next_preds, current_preds - MAX_ACC_DELTA, current_preds + MAX_ACC_DELTA)
        pred_lats[:, t] = next_preds
        
        # Update all contexts
        for i in range(N):
            sim_actions[i].append(action_sequences[i, t])
            sim_preds[i].append(next_preds[i])
            sim_states[i].append(State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t]
            ))
        current_preds = next_preds
    
    return pred_lats


def rollout_actions(model, action_seq, init_states, init_actions, init_preds, init_lataccel, future_plan, H):
    """
    Rollout action sequence through physics model.
    Reuses same logic as TinyPhysicsSimulator.sim_step but without full simulator overhead.
    
    Returns: predicted lateral accelerations for H steps
    """
    # Copy context (don't mutate caller's data)
    sim_states = list(init_states)
    sim_actions = list(init_actions)
    sim_preds = list(init_preds)
    
    pred_lat = np.zeros(H)
    current_pred = init_lataccel
    
    for t in range(H):
        # Model prediction (same as TinyPhysicsSimulator.sim_step)
        next_pred = model.get_current_lataccel(
            sim_states=sim_states[-CONTEXT_LENGTH:],
            actions=sim_actions[-CONTEXT_LENGTH:],
            past_preds=sim_preds[-CONTEXT_LENGTH:]
        )
        # Clip delta (same as TinyPhysicsSimulator.sim_step)
        next_pred = np.clip(next_pred, current_pred - MAX_ACC_DELTA, current_pred + MAX_ACC_DELTA)
        pred_lat[t] = next_pred
        
        # Update context with future state
        sim_actions.append(action_seq[t])
        sim_preds.append(next_pred)
        sim_states.append(State(
            roll_lataccel=future_plan.roll_lataccel[t],
            v_ego=future_plan.v_ego[t],
            a_ego=future_plan.a_ego[t]
        ))
        current_pred = next_pred
    
    return pred_lat

# Parallelization (ONNX releases GIL)
NUM_WORKERS = max(1, int(os.cpu_count() * 0.75))


class MPC_CEM(BaseController):
    """Model Predictive Control with Cross-Entropy Method optimization."""
    
    def __init__(self, model, control_horizon=15, prediction_horizon=50,
                 num_samples=150, num_elites=30, 
                 cem_iterations=3, action_std=0.15):
        super().__init__()
        self.model = model
        self.control_horizon = control_horizon    # Actions to optimize
        self.prediction_horizon = prediction_horizon  # Steps to evaluate
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.action_std = action_std
        
        # History for model context
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        
        # Diagnostics
        self.timestep = 0
        self.prev_mean = None  # Warm-start
        
        # Running cost tracking
        self.running_lat_cost = 0.0
        self.running_jerk_cost = 0.0
        self.running_steps = 0
        self.prev_lataccel = None
        
        # Reuse thread pool (don't recreate every iteration!)
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """MPC with CEM optimization."""
        self.timestep += 1
        
        # Track running cost (starts at step 100 per challenge rules)
        if self.timestep >= 100:
            lat_error = (target_lataccel - current_lataccel) ** 2
            self.running_lat_cost += lat_error
            
            jerk_val = 0.0
            if self.prev_lataccel is not None:
                jerk_val = (current_lataccel - self.prev_lataccel) / DEL_T
            self.running_jerk_cost += jerk_val ** 2
            self.running_steps += 1
            
            if self.timestep % 50 == 0:
                avg_lat = (self.running_lat_cost / self.running_steps) * 100 * LAT_ACCEL_COST_MULTIPLIER
                avg_jerk = (self.running_jerk_cost / self.running_steps) * 100
                total = avg_lat + avg_jerk
                print(f"  Step {self.timestep}: cost={total:.1f} (lat={avg_lat:.1f}, jerk={avg_jerk:.1f})", flush=True)
        elif self.timestep % 50 == 0:
            print(f"  Step {self.timestep}...", flush=True)
        
        self.prev_lataccel = current_lataccel
        
        H_predict = min(self.prediction_horizon, len(future_plan.lataccel))  # Steps to evaluate
        H_control = min(self.control_horizon, H_predict)  # Actions to optimize (can't exceed prediction!)
        
        # Fallback to P control if no history
        if H_predict == 0 or len(self.action_history) < CONTEXT_LENGTH:
            action = 0.195 * (target_lataccel - current_lataccel)
            self._update_history(state, current_lataccel, action)
            return action
        
        # Warm-start from previous solution (shift by 1)
        if self.prev_mean is not None and len(self.prev_mean) >= H_control:
            mean = np.concatenate([self.prev_mean[1:H_control], [0.0]])
        else:
            mean = np.zeros(H_control)
        
        std = np.ones(H_control) * self.action_std
        
        # CEM iterations
        for iter_num in range(self.cem_iterations):
            t_iter_start = time.time()
            
            # Sample sequences (only H_control actions)
            t_sample = time.time()
            sequences = np.random.randn(self.num_samples, H_control) * std + mean
            t_sample_elapsed = time.time() - t_sample
            
            # Evaluate using BATCHED rollout (much faster!)
            t_eval = time.time()
            # Extend all sequences to H_predict
            if H_predict > H_control:
                extended_seqs = np.column_stack([sequences, np.repeat(sequences[:, -1:], H_predict - H_control, axis=1)])
            else:
                extended_seqs = sequences
            
            # Batch rollout all sequences at once
            pred_lats = rollout_actions_batched(
                self.model, extended_seqs,
                self.state_history[-CONTEXT_LENGTH:],
                self.action_history[-CONTEXT_LENGTH:],
                self.pred_history[-CONTEXT_LENGTH:],
                current_lataccel, future_plan, H_predict
            )
            
            # Compute costs for all sequences
            target_seq = np.array(future_plan.lataccel[:H_predict])
            costs = np.array([compute_rollout_cost(target_seq, pred_lats[i]) for i in range(self.num_samples)])
            t_eval_elapsed = time.time() - t_eval
            
            # Keep elites, update distribution
            t_cem = time.time()
            elite_idx = np.argsort(costs)[:self.num_elites]
            elites = sequences[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6
            t_cem_elapsed = time.time() - t_cem
            
            t_iter_total = time.time() - t_iter_start
            if self.timestep in [100, 200, 300]:  # Log first few steps
                print(f"    CEM iter {iter_num+1}: total={t_iter_total:.2f}s (sample={t_sample_elapsed:.3f}s, eval={t_eval_elapsed:.2f}s, cem={t_cem_elapsed:.3f}s)", flush=True)
        
        # Execute first action, cache mean for next step
        best_action = float(mean[0])
        self.prev_mean = mean
        self._update_history(state, current_lataccel, best_action)
        return best_action
    
    def _eval_sequence(self, action_seq, future_plan, current_lat, H_control, H_predict):
        """Evaluate action sequence cost. After H_control actions, hold last action constant."""
        # Extend actions: [u0, ..., u_H, u_H, u_H, ...] if needed
        if H_predict > H_control:
            extended_actions = np.concatenate([action_seq, np.full(H_predict - H_control, action_seq[-1])])
        else:
            extended_actions = action_seq  # No extension needed
        
        # Rollout using shared logic (matches TinyPhysicsSimulator.sim_step)
        pred_lat = rollout_actions(
            self.model, extended_actions,
            self.state_history[-CONTEXT_LENGTH:],
            self.action_history[-CONTEXT_LENGTH:],
            self.pred_history[-CONTEXT_LENGTH:],
            current_lat, future_plan, H_predict
        )
        
        # Compute cost using EXACT official formula
        target_seq = np.array(future_plan.lataccel[:H_predict])
        return compute_rollout_cost(target_seq, pred_lat)
    
    def _update_history(self, state, lataccel, action):
        """Update history buffers."""
        self.state_history.append(state)
        self.pred_history.append(lataccel)
        self.action_history.append(action)


def add_batched_inference(model):
    """Add batched inference method to TinyPhysicsModel for speedup."""
    def get_current_lataccel_batch(self, batch_states, batch_actions, batch_preds):
        """
        Batched version: run ONNX once for all sequences.
        batch_states: list of [list of State]
        batch_actions: list of [list of float]
        batch_preds: list of [list of float]
        Returns: np.array of predictions [N]
        """
        N = len(batch_states)
        tokenized_actions = np.array([self.tokenizer.encode(preds) for preds in batch_preds])
        raw_states = np.array([[list(s) for s in states] for states in batch_states])
        actions_arr = np.array([list(actions) for actions in batch_actions])
        states = np.concatenate([actions_arr[:, :, np.newaxis], raw_states], axis=2)
        
        input_data = {
            'states': states.astype(np.float32),
            'tokens': tokenized_actions.astype(np.int64)
        }
        res = self.ort_session.run(None, input_data)[0]
        probs = self.softmax(res / 0.8, axis=-1)
        samples = np.array([np.random.choice(probs.shape[2], p=probs[i, -1]) for i in range(N)])
        return self.tokenizer.decode(samples)
    
    model.get_current_lataccel_batch = lambda *args: get_current_lataccel_batch(model, *args)


if __name__ == '__main__':
    from tinyphysics import TinyPhysicsSimulator
    
    print("=" * 60)
    print("MPC with CEM - CPU Optimized")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading ONNX model...", flush=True)
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    t_start = time.time()
    model = TinyPhysicsModel(str(model_path), debug=False)
    add_batched_inference(model)  # Add batched inference for speedup!
    print(f"      ✓ Model loaded with batched inference in {time.time()-t_start:.1f}s", flush=True)
    
    # Initialize controller
    print("\n[2/4] Initializing MPC controller...", flush=True)
    print(f"      Control horizon: 15 actions")
    print(f"      Prediction horizon: 50 steps (USE ALL FUTURE!)")
    print(f"      Samples: 150, Elites: 40, Iterations: 3")
    print(f"      Action std: 0.15")
    print(f"      Cost: PURE (lataccel*50 + jerk only, NO smoothness!)")
    print(f"      BATCHED INFERENCE: 50 model calls/step (was 22,500!)", flush=True)
    print(f"      Expected: ~5-8 min (3-5x faster than unbatched)", flush=True)
    # Pure MPC: Optimize EXACTLY the objective we're judged on
    controller = MPC_CEM(model, control_horizon=15, prediction_horizon=50,
                         num_samples=150, num_elites=40, 
                         cem_iterations=3, action_std=0.15)
    print(f"      ✓ Controller ready", flush=True)
    
    # Run simulation
    print("\n[3/4] Running simulation...", flush=True)
    print("      (First MPC call at step 20 may take 10-15s with batching)", flush=True)
    print("      Expected total time: ~5-8 min with batched ONNX", flush=True)
    print()
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    # Results
    print("\n[4/4] Complete!", flush=True)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"{'='*60}")
    print(f"  Total Cost: {cost['total_cost']:.2f}")
    print(f"    └─ Lataccel: {cost['lataccel_cost']:.2f} × 50 = {cost['lataccel_cost']*50:.1f}")
    print(f"    └─ Jerk:     {cost['jerk_cost']:.2f}")
    print(f"\n  Computation Time: {(t1-t0)/60:.1f} min")
    print(f"  Speedup vs baseline (36 min): {36/((t1-t0)/60):.1f}x")
    print(f"\n  Target: <100")
    if cost['total_cost'] < 100:
        print(f"  Status: ✓ SUCCESS!")
    else:
        print(f"  Status: ✗ Gap of {cost['total_cost']-100:.1f} to reach target")
    print(f"{'='*60}")
