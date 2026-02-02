"""
Exp045: Simple Model Predictive Control (MPC)

Key insight: The 50-step future is the OBJECTIVE, not the input.

MPC:
1. Samples action sequences [u0, u1, ..., u_H]
2. Rolls each sequence forward through physics model
3. Computes cost = tracking_error + jerk
4. Picks best sequence
5. Executes only u0, repeats (receding horizon)
"""

import numpy as np
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, BaseController, LAT_ACCEL_COST_MULTIPLIER, DEL_T

# Number of parallel workers (ONNX releases GIL, so threading works!)
NUM_WORKERS = max(1, int(os.cpu_count() * 0.75))
USE_PARALLEL = True  # Set to False to disable for debugging


class SimpleMPC(BaseController):
    """
    Simple MPC using random shooting.
    
    At each timestep:
    - Sample N action sequences
    - Roll each forward H steps
    - Pick the sequence with lowest cost
    - Execute first action only
    """
    
    def __init__(self, model, horizon=10, num_samples=1000, action_std=0.15):
        super().__init__()
        self.model = model
        self.horizon = horizon  # How many steps to plan ahead
        self.num_samples = num_samples  # How many sequences to try
        self.action_std = action_std  # Exploration noise (reduced for smoothness)
        
        # Track history for model rollouts
        from tinyphysics import CONTEXT_LENGTH
        self.CONTEXT_LENGTH = CONTEXT_LENGTH
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        
        # For debugging: track timestep
        self.timestep = 0
        
        # Warm-start: cache previous solution for next iteration
        self.prev_solution = None
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        MPC update: optimize action sequence, execute first action.
        
        Args:
            target_lataccel: Desired lateral accel at current step
            current_lataccel: Current lateral accel
            state: Current vehicle state
            future_plan: 50-step future (the TARGET to match + state evolution)
        
        Returns:
            Best steering action (first of optimized sequence)
        """
        self.timestep += 1
        if self.timestep % 50 == 0:
            print(f"  MPC step {self.timestep}...", flush=True)
        
        # How many future steps we can use (min of horizon and available future)
        H = min(self.horizon, len(future_plan.lataccel))
        
        if H == 0 or len(self.action_history) < self.CONTEXT_LENGTH:
            # No future available or not enough history, use simple feedback
            error = target_lataccel - current_lataccel
            action = 0.195 * error  # Simple P control
            # Update history AFTER computing action
            self.state_history.append(state)
            self.pred_history.append(current_lataccel)
            self.action_history.append(action)
            return action
        
        # Sample action sequences: [num_samples, H]
        # WARM-START: If we have previous solution, shift it and add noise
        if self.prev_solution is not None and len(self.prev_solution) > 1:
            # Shift previous solution: [u1, u2, ..., u_H, 0]
            warm_start = np.concatenate([self.prev_solution[1:], [0.0]])
            if len(warm_start) < H:
                warm_start = np.pad(warm_start, (0, H - len(warm_start)))
            elif len(warm_start) > H:
                warm_start = warm_start[:H]
            # Sample around warm start
            action_sequences = np.random.randn(self.num_samples, H) * self.action_std + warm_start
        else:
            # Cold start: sample from zero mean
            action_sequences = np.random.randn(self.num_samples, H) * self.action_std
        
        # Evaluate each sequence using REAL model (PARALLELIZED!)
        if USE_PARALLEL and self.num_samples >= 20:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                eval_args = [
                    (seq, future_plan, current_lataccel, state,
                     self.state_history, self.action_history, self.pred_history, H)
                    for seq in action_sequences
                ]
                costs = np.array(list(executor.map(
                    lambda args: self._evaluate_sequence(*args),
                    eval_args
                )))
        else:
            # Sequential fallback
            costs = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                costs[i] = self._evaluate_sequence(
                    action_sequences[i], 
                    future_plan,  # Pass full future_plan (includes target + states)
                    current_lataccel,
                    state,
                    self.state_history,
                    self.action_history,
                    self.pred_history,
                    H  # Pass horizon
                )
        
        # Pick best sequence
        best_idx = np.argmin(costs)
        best_sequence = action_sequences[best_idx]
        
        # Cache for next iteration
        self.prev_solution = best_sequence.copy()
        
        # Execute only first action (receding horizon)
        best_action = float(best_sequence[0])
        
        # Update history AFTER computing action
        self.state_history.append(state)
        self.pred_history.append(current_lataccel)
        self.action_history.append(best_action)
        
        return best_action
    
    def _evaluate_sequence(self, action_seq, future_plan, current_lat, state, 
                          sim_states, past_actions, past_preds, H):
        """
        Roll action sequence forward using REAL TinyPhysics model, compute cost.
        
        Cost = tracking_error + jerk (matching leaderboard cost)
        """
        from tinyphysics import CONTEXT_LENGTH, MAX_ACC_DELTA, State
        
        # Extract target trajectory and future states
        target_lat_seq = np.array(future_plan.lataccel[:H])
        
        # Initialize simulation history (need CONTEXT_LENGTH for model)
        sim_state_hist = list(sim_states[-CONTEXT_LENGTH:])
        sim_action_hist = list(past_actions[-CONTEXT_LENGTH:])
        sim_pred_hist = list(past_preds[-CONTEXT_LENGTH:])
        
        # Simulate forward H steps using REAL model with REAL future states
        pred_lat = np.zeros(H)
        current_pred = current_lat
        
        for t in range(H):
            # Get prediction from real TinyPhysics model
            next_pred = self.model.get_current_lataccel(
                sim_states=sim_state_hist[-CONTEXT_LENGTH:],
                actions=sim_action_hist[-CONTEXT_LENGTH:],
                past_preds=sim_pred_hist[-CONTEXT_LENGTH:]
            )
            
            # Clip delta (same as simulator)
            next_pred = np.clip(next_pred, current_pred - MAX_ACC_DELTA, current_pred + MAX_ACC_DELTA)
            
            pred_lat[t] = next_pred
            
            # Update history for next step
            sim_action_hist.append(action_seq[t])
            sim_pred_hist.append(next_pred)
            
            # ✓ FIXED: Use ACTUAL future state from future_plan
            future_state = State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t]
            )
            sim_state_hist.append(future_state)
            
            current_pred = next_pred
        
        # Tracking error cost (matches README.md formula)
        tracking_error = pred_lat - target_lat_seq
        lat_cost = np.mean(tracking_error ** 2) * 100 * LAT_ACCEL_COST_MULTIPLIER
        
        # Jerk cost (smoothness of lataccel)
        jerk = np.diff(pred_lat, prepend=current_lat) / DEL_T
        jerk_cost = np.mean(jerk ** 2) * 100
        
        # Action smoothness cost (CRITICAL for reducing jerk!)
        # Penalize rapid changes in steering action
        if len(past_actions) > 0:
            action_changes = np.diff(action_seq, prepend=past_actions[-1])
        else:
            action_changes = np.diff(action_seq, prepend=0.0)
        action_smooth_cost = np.mean(action_changes ** 2) * 1000  # BASELINE (proven)
        
        return lat_cost + jerk_cost + action_smooth_cost


class SimpleMPC_WithCEM(BaseController):
    """
    MPC with Cross-Entropy Method (CEM) for better optimization.
    
    CEM:
    1. Sample N sequences from distribution
    2. Keep top K elite sequences
    3. Fit new distribution to elites
    4. Repeat for M iterations
    """
    
    def __init__(self, model, horizon=10, num_samples=500, num_elites=50, 
                 cem_iterations=3, action_std=0.15):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.action_std = action_std  # Very low for ultra-smooth actions
        
        # Track history for model rollouts
        from tinyphysics import CONTEXT_LENGTH
        self.CONTEXT_LENGTH = CONTEXT_LENGTH
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        
        # For debugging: track timestep and running cost
        self.timestep = 0
        self.running_lat_cost = 0.0
        self.running_jerk_cost = 0.0
        self.prev_lataccel = None
        
        # Warm-start: cache previous mean for initialization
        self.prev_mean = None
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """MPC with CEM optimization"""
        self.timestep += 1
        
        # Track running cost (matching simulator's cost function)
        if self.timestep >= 100:  # Only count after CONTROL_START_IDX
            lat_error = (target_lataccel - current_lataccel) ** 2
            self.running_lat_cost += lat_error
            
            if self.prev_lataccel is not None:
                jerk = ((current_lataccel - self.prev_lataccel) / DEL_T) ** 2
                self.running_jerk_cost += jerk
        
        self.prev_lataccel = current_lataccel
        
        # Print diagnostics
        if self.timestep % 50 == 0:
            steps_so_far = max(1, self.timestep - 100 + 1)
            avg_lat = (self.running_lat_cost / steps_so_far) * 100 * LAT_ACCEL_COST_MULTIPLIER
            avg_jerk = (self.running_jerk_cost / max(1, steps_so_far - 1)) * 100
            total = avg_lat + avg_jerk
            print(f"  Step {self.timestep}: running_cost={total:.1f} (lat={avg_lat:.1f}, jerk={avg_jerk:.1f})", flush=True)
        
        H = min(self.horizon, len(future_plan.lataccel))
        
        if H == 0 or len(self.action_history) < self.CONTEXT_LENGTH:
            error = target_lataccel - current_lataccel
            action = 0.195 * error
            # Update history AFTER computing action
            self.state_history.append(state)
            self.pred_history.append(current_lataccel)
            self.action_history.append(action)
            return action
        
        # Initialize distribution with WARM-START
        if self.prev_mean is not None and len(self.prev_mean) > 1:
            # Shift previous mean: [u1, u2, ..., u_H, 0]
            mean = np.concatenate([self.prev_mean[1:], [0.0]])
            if len(mean) < H:
                mean = np.pad(mean, (0, H - len(mean)))
            elif len(mean) > H:
                mean = mean[:H]
        else:
            mean = np.zeros(H)
        std = np.ones(H) * self.action_std
        
        # CEM iterations
        for _ in range(self.cem_iterations):
            # Sample from current distribution
            action_sequences = np.random.randn(self.num_samples, H) * std + mean
            
            # Evaluate all sequences using REAL model (PARALLELIZED!)
            if USE_PARALLEL and self.num_samples >= 20:
                # Parallel evaluation using ThreadPoolExecutor
                # ONNX releases GIL, so threads work well here!
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    eval_args = [
                        (seq, future_plan, current_lataccel, state,
                         self.state_history, self.action_history, self.pred_history, H)
                        for seq in action_sequences
                    ]
                    costs = np.array(list(executor.map(
                        lambda args: self._evaluate_sequence(*args),
                        eval_args
                    )))
            else:
                # Sequential evaluation (fallback)
                costs = np.array([
                    self._evaluate_sequence(
                        seq, future_plan, current_lataccel, state,
                        self.state_history, self.action_history, self.pred_history, H
                    )
                    for seq in action_sequences
                ])
            
            # Select elite sequences (top K)
            elite_indices = np.argsort(costs)[:self.num_elites]
            elite_sequences = action_sequences[elite_indices]
            
            # Fit new distribution to elites
            mean = elite_sequences.mean(axis=0)
            std = elite_sequences.std(axis=0) + 1e-6  # Avoid collapse
        
        # Cache for next iteration
        self.prev_mean = mean.copy()
        
        # Return first action of best elite sequence
        best_action = float(mean[0])
        
        # Update history AFTER computing action
        self.state_history.append(state)
        self.pred_history.append(current_lataccel)
        self.action_history.append(best_action)
        
        return best_action
    
    def _evaluate_sequence(self, action_seq, future_plan, current_lat, state,
                          sim_states, past_actions, past_preds, H):
        """Same as SimpleMPC - use REAL TinyPhysics model with REAL future states"""
        from tinyphysics import CONTEXT_LENGTH, MAX_ACC_DELTA, State
        
        # Extract target trajectory and future states
        target_lat_seq = np.array(future_plan.lataccel[:H])
        
        # Initialize simulation history
        sim_state_hist = list(sim_states[-CONTEXT_LENGTH:])
        sim_action_hist = list(past_actions[-CONTEXT_LENGTH:])
        sim_pred_hist = list(past_preds[-CONTEXT_LENGTH:])
        
        # Simulate forward H steps using REAL model with REAL future states
        pred_lat = np.zeros(H)
        current_pred = current_lat
        
        for t in range(H):
            # Get prediction from real TinyPhysics model
            next_pred = self.model.get_current_lataccel(
                sim_states=sim_state_hist[-CONTEXT_LENGTH:],
                actions=sim_action_hist[-CONTEXT_LENGTH:],
                past_preds=sim_pred_hist[-CONTEXT_LENGTH:]
            )
            
            # Clip delta
            next_pred = np.clip(next_pred, current_pred - MAX_ACC_DELTA, current_pred + MAX_ACC_DELTA)
            
            pred_lat[t] = next_pred
            
            # Update history
            sim_action_hist.append(action_seq[t])
            sim_pred_hist.append(next_pred)
            
            # ✓ FIXED: Use ACTUAL future state from future_plan
            future_state = State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t]
            )
            sim_state_hist.append(future_state)
            
            current_pred = next_pred
        
        # Cost computation (matches README.md formula)
        tracking_error = pred_lat - target_lat_seq
        lat_cost = np.mean(tracking_error ** 2) * 100 * LAT_ACCEL_COST_MULTIPLIER
        
        jerk = np.diff(pred_lat, prepend=current_lat) / DEL_T
        jerk_cost = np.mean(jerk ** 2) * 100
        
        # Action smoothness cost (CRITICAL for reducing jerk!)
        if len(past_actions) > 0:
            action_changes = np.diff(action_seq, prepend=past_actions[-1])
        else:
            action_changes = np.diff(action_seq, prepend=0.0)
        action_smooth_cost = np.mean(action_changes ** 2) * 1000  # BASELINE (proven)
        
        return lat_cost + jerk_cost + action_smooth_cost


if __name__ == '__main__':
    # Quick test with REDUCED parameters for speed
    from tinyphysics import TinyPhysicsSimulator
    import time
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    
    # BASELINE + PARALLELIZATION ONLY
    # Use EXACT params that got 157 cost, just add parallelization
    print("Testing SimpleMPC_WithCEM - BASELINE + PARALLEL")
    print("  Using EXACT parameters that achieved 157 cost baseline")
    print("  Horizon: 12 steps = 1.2 seconds look-ahead")
    print("  Samples: 150, Elites: 30, Iterations: 4")
    print("  Action smoothness weight: 1000 (PROVEN to work)")
    print("  Action std: 0.15 (DEFAULT - proven)")
    print(f"  Parallelization: {'ENABLED' if USE_PARALLEL else 'DISABLED'}")
    print(f"  Worker threads: {NUM_WORKERS} (of {os.cpu_count()} CPUs)")
    print("  Strategy: DON'T CHANGE WHAT WORKS - just parallelize it!")
    print("  Expected: 157 cost in ~10 min (vs 36 min sequential)")
    print()
    t0 = time.time()
    controller_cem = SimpleMPC_WithCEM(model, horizon=12, num_samples=150, num_elites=30, cem_iterations=4)
    sim_cem = TinyPhysicsSimulator(model, str(data_path), controller=controller_cem, debug=False)
    cost_dict_cem = sim_cem.rollout()
    t1 = time.time()
    print(f"\n{'='*60}")
    print(f"FINAL RESULT (BASELINE PARAMS + PARALLEL):")
    print(f"  Total cost: {cost_dict_cem['total_cost']:.2f}")
    print(f"    - Lataccel cost: {cost_dict_cem['lataccel_cost']:.2f} × 50 = {cost_dict_cem['lataccel_cost']*50:.1f}")
    print(f"    - Jerk cost: {cost_dict_cem['jerk_cost']:.2f}")
    print(f"  Computation time: {t1-t0:.1f}s ({(t1-t0)/60:.1f} min)")
    print(f"  ")
    print(f"  Baseline (sequential): 157 cost in 36 min")
    print(f"  This run (parallel):   {cost_dict_cem['total_cost']:.0f} cost in {t1-t0/60:.1f} min")
    print(f"  Speedup: {36/(t1-t0/60):.1f}x faster")
    print(f"{'='*60}")
    
    if cost_dict_cem['total_cost'] < 100:
        print("\n🎉 SUCCESS! Cost < 100 achieved with MPC!")
        print(f"   Improvement over PID (~120): {120 - cost_dict_cem['total_cost']:.1f} points")
    else:
        gap = cost_dict_cem['total_cost'] - 100
        print(f"\n❌ Need to improve by {gap:.1f} to reach <100")
        print(f"   Target breakdown: ~50 (lataccel) + ~50 (jerk) = 100")
        print(f"   Current breakdown: {cost_dict_cem['lataccel_cost']*50:.1f} (lataccel) + {cost_dict_cem['jerk_cost']:.2f} (jerk) = {cost_dict_cem['total_cost']:.2f}")
        print(f"\n   Next steps to try:")
        print(f"   - Fine-tune smoothness weight (currently 3500, try 3000-4000)")
        print(f"   - Adjust std for balance (currently 0.12, try 0.10-0.14)")
        print(f"   - Try longer horizon (currently 15, could go to 18-20)")
        print(f"   - Increase CEM iterations (currently 3, try 4)")
        print(f"   - More samples (currently 150, max ~200 with parallel)")

