"""
Gymnasium wrapper for TinyPhysics with PID residual control
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel


class SimplePIDController:
    """Inline PID controller to avoid import issues in multiprocessing"""
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff


class DummyController:
    """Dummy controller that returns the pre-set action"""
    def __init__(self):
        self.action = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.action


class TinyPhysicsResidualEnv(gym.Env):
    """
    Single-file environment for residual PPO on top of PID baseline.
    
    State: 54D [error, error_rate, v_ego, a_ego, curvatures[50]]
    Action: 1D residual steering ∈ [-0.5, 0.5]
    Final action: PID + residual
    """
    
    def __init__(self, data_file):
        super().__init__()
        
        self.data_file = data_file
        # Store path for lazy initialization
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/tinyphysics.onnx"))
        self.model = None  # Lazy init in reset (for multiprocessing)
        self.pid = None  # Will be initialized in reset()
        
        # Gym spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        
        # Episode tracking
        self.sim = None
        self.step_idx = 0
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.episode_costs = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Lazy init model (for multiprocessing compatibility)
        if self.model is None:
            self.model = TinyPhysicsModel(self.model_path, debug=False)
        
        # Create dummy controller for simulator (we'll set actions manually)
        dummy_controller = DummyController()
        
        # Create new simulator
        self.sim = TinyPhysicsSimulator(self.model, str(self.data_file), controller=dummy_controller, debug=False)
        self.dummy_controller = dummy_controller  # Keep reference
        self.step_idx = 0
        self.prev_error = 0.0
        self.error_integral = 0.0
        self.episode_costs = []
        
        # Reset PID (create new instance)
        self.pid = SimplePIDController()
        
        # Get initial state
        state = self._get_state()
        return state, {}
    
    def _get_state(self):
        """Build 54D state vector with curvatures"""
        if self.step_idx >= len(self.sim.data):
            # Episode ended, return zeros
            return np.zeros(54, dtype=np.float32)
        
        # Get current state and future plan
        sim_state, target_lataccel, future_plan = self.sim.get_state_target_futureplan(self.step_idx)
        
        # Compute error and derivatives
        current_lataccel = self.sim.current_lataccel_history[self.step_idx] if self.step_idx < len(self.sim.current_lataccel_history) else 0.0
        error = target_lataccel - current_lataccel
        error_rate = error - self.prev_error
        self.prev_error = error
        self.error_integral += error
        
        # Get velocity and acceleration
        v_ego = sim_state.v_ego
        a_ego = sim_state.a_ego
        
        # Convert future lat accels to curvatures
        # curvature = lat_accel / v²
        v_ego_sq = max(v_ego ** 2, 25.0)  # clamp to min 5 m/s to avoid division by zero
        
        future_lataccels = np.array(future_plan.lataccel, dtype=np.float32)
        if len(future_lataccels) < 50:
            # Pad if needed
            future_lataccels = np.pad(future_lataccels, (0, 50 - len(future_lataccels)), mode='edge')
        else:
            future_lataccels = future_lataccels[:50]
        
        curvatures = future_lataccels / v_ego_sq
        
        # Normalize state components
        state = np.array([
            error / 2.0,           # [0] tracking error
            error_rate / 1.0,      # [1] error rate
            v_ego / 30.0,          # [2] velocity
            a_ego / 3.0,           # [3] acceleration
            *curvatures / 0.5,     # [4:54] curvatures (normalized)
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Execute one step with residual action"""
        # Check if episode is done (need buffer for future plan)
        if self.step_idx >= len(self.sim.data) - 50 - 1:
            # Episode ending, return terminal state
            return np.zeros(54, dtype=np.float32), 0.0, True, False, {'episode_cost': sum(self.episode_costs)}
        
        # Get current state for PID
        sim_state, target_lataccel, future_plan = self.sim.get_state_target_futureplan(self.step_idx)
        current_lataccel = self.sim.current_lataccel_history[self.step_idx] if self.step_idx < len(self.sim.current_lataccel_history) else 0.0
        
        # PID action (needs target, current, state, future_plan)
        pid_action = self.pid.update(target_lataccel, current_lataccel, sim_state, future_plan)
        
        # Residual action (bounded)
        residual_action = np.clip(action[0], -0.5, 0.5)
        
        # Final action
        final_action = np.clip(pid_action + residual_action, -2.0, 2.0)
        
        # Set action in dummy controller (simulator will call it)
        self.dummy_controller.action = final_action
        
        # Execute step (simulator will use dummy controller)
        self.sim.step()
        self.step_idx += 1
        
        # Compute costs for this step (matching tinyphysics.py)
        if self.step_idx > 0 and self.step_idx < len(self.sim.data):
            # Lateral acceleration cost (squared error)
            target = self.sim.target_lataccel_history[self.step_idx - 1]
            pred = self.sim.current_lataccel_history[self.step_idx - 1]
            lat_cost = (target - pred) ** 2 * 100
            
            # Jerk cost (squared derivative)
            if self.step_idx > 1:
                prev_pred = self.sim.current_lataccel_history[self.step_idx - 2]
                curr_pred = self.sim.current_lataccel_history[self.step_idx - 1]
                DEL_T = 0.1  # From tinyphysics.py
                jerk_cost = ((curr_pred - prev_pred) / DEL_T) ** 2 * 100
            else:
                jerk_cost = 0.0
            
            step_cost = 50 * lat_cost + jerk_cost
            self.episode_costs.append(step_cost)
        else:
            step_cost = 0.0
        
        # Reward (negative cost, normalized)
        reward = -step_cost / 100.0
        
        # Check if done (need buffer for future plan)
        done = self.step_idx >= len(self.sim.data) - 50 - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {}
        if done:
            info['episode_cost'] = sum(self.episode_costs)
        
        return next_state, reward, done, False, info


def make_env(data_files):
    """Create vectorized environment with multiple data files"""
    def _make_single_env(data_file):
        def _init():
            return TinyPhysicsResidualEnv(data_file)
        return _init
    
    env_fns = [_make_single_env(f) for f in data_files]
    return gym.vector.AsyncVectorEnv(env_fns)
