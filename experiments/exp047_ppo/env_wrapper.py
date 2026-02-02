"""
Wrap the ACTUAL TinyPhysicsSimulator - don't reimplement
This MUST match exactly since we're using the same code
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController, CONTROL_START_IDX


class EnvController(BaseController):
    """Wrapper that lets RL agent control through update()"""
    def __init__(self):
        self.next_action = None
        self.last_obs = None
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Save observation
        self.last_obs = {
            'target': target_lataccel,
            'current': current_lataccel,
            'state': state,
            'futureplan': future_plan
        }
        
        # Return action set by RL agent
        if self.next_action is None:
            return 0.0  # Fallback
        return self.next_action


class SimulatorEnv:
    """Wraps TinyPhysicsSimulator for RL"""
    
    def __init__(self, model_path, data_path):
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.data_path = data_path
        self.controller = EnvController()
        self.sim = None
        
    def reset(self):
        """Start new episode"""
        self.sim = TinyPhysicsSimulator(self.model, self.data_path, self.controller, debug=False)
        # After init, sim has histories up to CONTEXT_LENGTH
        # Next step will be at CONTEXT_LENGTH
        return self._get_obs()
    
    def _get_obs(self):
        """Get observation - called BEFORE step()"""
        # Observation uses data that controller.update() will see
        if self.sim.step_idx >= len(self.sim.data):
            return None
        
        # Get what the controller will see
        state, target, futureplan = self.sim.get_state_target_futureplan(self.sim.step_idx)
        
        return {
            'target': target,
            'current': self.sim.current_lataccel,
            'state': state,
            'futureplan': futureplan,
            'step': self.sim.step_idx
        }
    
    def step(self, action):
        """Take action"""
        # Set action for controller
        self.controller.next_action = float(action)
        
        # Step simulator
        self.sim.step()
        
        # Check if done
        done = (self.sim.step_idx >= len(self.sim.data))
        
        # Get next obs
        obs = self._get_obs() if not done else None
        
        # Reward (computed at end)
        reward = 0.0
        
        return obs, reward, done
    
    def compute_cost(self):
        """Use simulator's exact cost computation"""
        return self.sim.compute_cost()


if __name__ == '__main__':
    # Test with PID
    class PIDController:
        def __init__(self):
            self.p = 0.195
            self.i = 0.100
            self.d = -0.053
            self.error_integral = 0
            self.prev_error = 0
        
        def act(self, obs):
            error = obs['target'] - obs['current']
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            return self.p * error + self.i * self.error_integral + self.d * error_diff
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("Testing wrapped TinyPhysicsSimulator with PID...")
    print("="*60)
    
    env = SimulatorEnv(str(model_path), str(data_path))
    controller = PIDController()
    
    obs = env.reset()
    done = False
    
    while not done:
        action = controller.act(obs)
        obs, reward, done = env.step(action)
    
    cost = env.compute_cost()
    
    print(f"Wrapped Simulator:")
    print(f"  Lat cost:   {cost['lataccel_cost']:.4f}")
    print(f"  Jerk cost:  {cost['jerk_cost']:.4f}")
    print(f"  Total cost: {cost['total_cost']:.4f}")
    
    print(f"\nOfficial (tinyphysics.py --controller pid):")
    print(f"  Lat cost:   1.293")
    print(f"  Jerk cost:  35.56")
    print(f"  Total cost: 100.2")
    
    print(f"\n{'='*60}")
    diff = abs(cost['total_cost'] - 100.2)
    if diff < 0.01:
        print(f"✓ EXACT MATCH (diff: {diff:.6f})")
    else:
        print(f"✗ Difference: {diff:.4f}")
