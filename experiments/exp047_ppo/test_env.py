"""
Test that our environment matches official simulator
Run PID through it, check cost matches
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from env_simple import SimpleEnv
from tinyphysics import LAT_ACCEL_COST_MULTIPLIER, DEL_T, CONTROL_START_IDX


class PIDController:
    def __init__(self):
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
    
    def act(self, obs):
        error = obs['error']
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff


if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("Testing environment with PID...")
    print("="*60)
    
    env = SimpleEnv(str(model_path), str(data_path))
    controller = PIDController()
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    # Collect trajectory
    errors = []
    lataccels = []
    
    while not done:
        action = controller.act(obs)
        obs, reward, done = env.step(action)
        total_reward += reward
        
        if env.step_idx > CONTROL_START_IDX:
            errors.append(obs['error'] if obs else 0)
            lataccels.append(env.current_lataccel)
    
    # Compute cost (like official simulator)
    errors = np.array(errors)
    lataccels = np.array(lataccels)
    
    lat_cost = np.mean(errors ** 2) * 100
    jerk_cost = np.mean((np.diff(lataccels) / DEL_T) ** 2) * 100
    total_cost = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost
    
    print(f"\nOur Environment:")
    print(f"  Lat cost:   {lat_cost:.2f}")
    print(f"  Jerk cost:  {jerk_cost:.2f}")
    print(f"  Total cost: {total_cost:.2f}")
    
    print(f"\nOfficial (from tinyphysics.py --controller pid):")
    print(f"  Total cost: 100.2")
    
    print(f"\n{'='*60}")
    if abs(total_cost - 100.2) < 5:
        print("✓ Environment matches official!")
    else:
        print(f"✗ Mismatch: {abs(total_cost - 100.2):.1f} point difference")
