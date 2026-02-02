"""
MPC v2: Start from PID, refine locally

Strategy: Use PID as baseline, search small perturbations
This ensures temporal consistency (actions don't jump around)
"""

import numpy as np
from . import BaseController
from tinyphysics import LAT_ACCEL_COST_MULTIPLIER, DEL_T, CONTEXT_LENGTH, MAX_ACC_DELTA, State, STEER_RANGE


class Controller(BaseController):
    """PID + Local MPC Search"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # MPC params (small, fast)
        self.H = 10              # Short horizon
        self.samples = 30        # Few samples
        self.search_radius = 0.3 # Search around PID
        
        # History
        self.state_history = []
        self.action_history = []
        self.pred_history = []
        self.model = None
        
        np.random.seed(42)
        
    def set_model(self, model):
        self.model = model
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Get PID baseline
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        pid_action = self.p * error + self.i * self.error_integral + self.d * error_diff
        pid_action = np.clip(pid_action, STEER_RANGE[0], STEER_RANGE[1])
        
        # Update history first
        self.state_history.append(state)
        self.pred_history.append(current_lataccel)
        self.action_history.append(pid_action)
        
        # If no model or future, just use PID
        if (self.model is None or 
            len(future_plan.lataccel) < self.H or 
            len(self.state_history) < CONTEXT_LENGTH):
            return float(pid_action)
        
        # Try small perturbations around PID
        best_action = pid_action
        best_cost = float('inf')
        
        # Sample actions near PID
        perturbations = np.random.randn(self.samples) * self.search_radius
        actions = np.clip(pid_action + perturbations, STEER_RANGE[0], STEER_RANGE[1])
        
        for action in actions:
            cost = self._eval_1step(action, future_plan.lataccel[0], current_lataccel)
            if cost < best_cost:
                best_cost = cost
                best_action = action
        
        # Update last action in history with refined one
        self.action_history[-1] = best_action
        return float(best_action)
    
    def _eval_1step(self, action, target_next, current_lat):
        """Simple 1-step lookahead"""
        # Predict next state
        next_pred = self.model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.pred_history[-CONTEXT_LENGTH:]
        )
        next_pred = np.clip(next_pred, current_lat - MAX_ACC_DELTA, current_lat + MAX_ACC_DELTA)
        
        # Cost: tracking error + jerk
        track_error = (next_pred - target_next) ** 2
        jerk = ((next_pred - current_lat) / DEL_T) ** 2
        
        return track_error * LAT_ACCEL_COST_MULTIPLIER + jerk
