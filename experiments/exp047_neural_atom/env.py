"""
Gym wrapper for TinyPhysics - exactly matches tinyphysics.py
"""

import gymnasium as gym
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController
from tinyphysics import CONTROL_START_IDX, COST_END_IDX, DEL_T, LAT_ACCEL_COST_MULTIPLIER


class DummyController(BaseController):
    """Controller that takes actions from RL agent"""
    def __init__(self):
        self.action = 0.0
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.action


class TinyPhysicsEnv(gym.Env):
    """Single trajectory environment"""
    
    def __init__(self, model_path, data_path):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        
        # Observation: [error, error_integral, error_diff, roll_lataccel]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        # Action: steer command [-2, 2]
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )
        
        self.model = TinyPhysicsModel(str(model_path), debug=False)
        self.controller = DummyController()
        self.sim = None
        
        # State for PID features
        self.error_integral = 0.0
        self.prev_error = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Create new simulator
        self.controller = DummyController()
        self.sim = TinyPhysicsSimulator(
            self.model, str(self.data_path), 
            controller=self.controller, debug=False
        )
        
        # Reset PID state
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        # Get initial observation
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        """Get current observation: [error, error_integral, error_diff, roll_lataccel]"""
        # Use last available values
        target = self.sim.target_lataccel_history[-1]
        current = self.sim.current_lataccel
        state = self.sim.state_history[-1]
        
        error = target - current
        error_diff = error - self.prev_error
        
        return np.array([
            error, 
            self.error_integral, 
            error_diff, 
            state.roll_lataccel
        ], dtype=np.float32)
    
    def _get_reward(self, prev_lataccel, curr_lataccel, target_lataccel):
        """Per-step reward"""
        # Tracking error
        tracking_error = (target_lataccel - curr_lataccel) ** 2
        
        # Jerk penalty
        jerk = (curr_lataccel - prev_lataccel) / DEL_T
        jerk_penalty = jerk ** 2
        
        # Negative cost (to maximize)
        reward = -(tracking_error * LAT_ACCEL_COST_MULTIPLIER + jerk_penalty) * 0.01
        
        return reward
    
    def step(self, action):
        # Store previous lataccel for reward
        prev_lataccel = self.sim.current_lataccel
        
        # Set action
        self.controller.action = float(action[0])
        
        # Step simulator
        self.sim.step()
        
        # Update PID state
        target = self.sim.target_lataccel_history[self.sim.step_idx - 1]
        current = self.sim.current_lataccel_history[self.sim.step_idx - 1]
        error = target - current
        self.error_integral += error
        self.prev_error = error
        
        # Get reward
        target_now = self.sim.target_lataccel_history[self.sim.step_idx - 1]
        reward = self._get_reward(prev_lataccel, self.sim.current_lataccel, target_now)
        
        # Check if done
        done = self.sim.step_idx >= len(self.sim.data)
        terminated = done
        truncated = False
        
        # Get next observation
        if not done:
            obs = self._get_obs()
        else:
            obs = np.zeros(4, dtype=np.float32)
        
        # Include official cost in info when episode ends
        info = {}
        if done:
            info['official_cost'] = self.sim.compute_cost()['total_cost']
        
        return obs, reward, terminated, truncated, info
