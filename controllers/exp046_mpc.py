"""
Clean MPC Controller
Target: Beat PID (84.85) on batch metrics
"""

import numpy as np
from . import BaseController
from tinyphysics import LAT_ACCEL_COST_MULTIPLIER, DEL_T, CONTEXT_LENGTH, MAX_ACC_DELTA, State, STEER_RANGE
from concurrent.futures import ThreadPoolExecutor
import os


class Controller(BaseController):
    """Simple MPC with CEM"""
    
    def __init__(self):
        # MPC params (aggressive for quality)
        self.H = 30              # Horizon: 3 seconds
        self.samples = 100       # CEM samples
        self.elites = 25         # Top performers
        self.iters = 4           # CEM iterations
        self.std = 0.15          # Exploration std
        
        # State tracking
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        self.prev_mean = None
        
        # Model (need access to physics)
        self.model = None
        
        # Parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(os.cpu_count() * 0.75)))
        
        np.random.seed(42)
        
    def set_model(self, model):
        """Called by simulator to provide model access"""
        self.model = model
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Initialize history
        if len(self.state_history) == 0:
            self.state_history.append(state)
            self.pred_history.append(current_lataccel)
            # Fallback for first step
            action = 0.195 * (target_lataccel - current_lataccel)
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.action_history.append(action)
            return float(action)
        
        # Check if we have model and future
        if self.model is None or len(future_plan.lataccel) == 0:
            # PID fallback
            action = 0.195 * (target_lataccel - current_lataccel)
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.state_history.append(state)
            self.pred_history.append(current_lataccel)
            self.action_history.append(action)
            return float(action)
        
        # MPC
        H = min(self.H, len(future_plan.lataccel))
        
        # Warm-start mean
        if self.prev_mean is not None and len(self.prev_mean) >= H:
            mean = np.concatenate([self.prev_mean[1:H], [0.0]])
        else:
            mean = np.zeros(H)
        std = np.ones(H) * self.std
        
        # CEM optimization
        for _ in range(self.iters):
            # Sample sequences
            seqs = np.random.randn(self.samples, H) * std + mean
            seqs = np.clip(seqs, STEER_RANGE[0], STEER_RANGE[1])
            
            # Evaluate in parallel
            costs = np.array(list(self.executor.map(
                lambda seq: self._eval(seq, future_plan, current_lataccel, H),
                seqs
            )))
            
            # Select elites
            elite_idx = np.argsort(costs)[:self.elites]
            elites = seqs[elite_idx]
            
            # Update distribution
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6
        
        # Execute first action
        action = float(np.clip(mean[0], STEER_RANGE[0], STEER_RANGE[1]))
        self.prev_mean = mean
        
        # Update history
        self.state_history.append(state)
        self.pred_history.append(current_lataccel)
        self.action_history.append(action)
        
        return action
    
    def _eval(self, actions, future_plan, current_lat, H):
        """Evaluate action sequence"""
        # Get context
        if len(self.state_history) < CONTEXT_LENGTH:
            # Not enough history, return high cost
            return 1e6
        
        states = list(self.state_history[-CONTEXT_LENGTH:])
        sim_actions = list(self.action_history[-CONTEXT_LENGTH:])
        preds = list(self.pred_history[-CONTEXT_LENGTH:])
        
        # Rollout
        pred_lat = []
        curr = current_lat
        
        for t in range(H):
            # Predict next lateral acceleration
            next_pred = self.model.get_current_lataccel(
                sim_states=states[-CONTEXT_LENGTH:],
                actions=sim_actions[-CONTEXT_LENGTH:],
                past_preds=preds[-CONTEXT_LENGTH:]
            )
            next_pred = np.clip(next_pred, curr - MAX_ACC_DELTA, curr + MAX_ACC_DELTA)
            pred_lat.append(next_pred)
            
            # Update context
            sim_actions.append(actions[t])
            preds.append(next_pred)
            states.append(State(
                roll_lataccel=future_plan.roll_lataccel[t],
                v_ego=future_plan.v_ego[t],
                a_ego=future_plan.a_ego[t]
            ))
            curr = next_pred
        
        # Compute cost (EXACT match to tinyphysics.py)
        target = np.array(future_plan.lataccel[:H])
        pred = np.array(pred_lat)
        
        lat_cost = np.mean((target - pred) ** 2) * 100 * LAT_ACCEL_COST_MULTIPLIER
        jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
        
        return lat_cost + jerk_cost
