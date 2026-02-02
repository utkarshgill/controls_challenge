"""
Diagnostic controller: Log inputs to understand the data
"""

import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import BaseController, STEER_RANGE


class Controller(BaseController):
    """PID with extensive logging"""
    
    def __init__(self):
        # PID params
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
        # Diagnostics
        self.step = 0
        self.logs = defaultdict(list)
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step += 1
        
        # Log scalars
        self.logs['target_lataccel'].append(target_lataccel)
        self.logs['current_lataccel'].append(current_lataccel)
        
        # Log state
        self.logs['state_roll_lataccel'].append(state.roll_lataccel)
        self.logs['state_v_ego'].append(state.v_ego)
        self.logs['state_a_ego'].append(state.a_ego)
        
        # Log future_plan metadata
        self.logs['future_plan_len'].append(len(future_plan.lataccel))
        
        # Log first few elements of future_plan arrays
        if len(future_plan.lataccel) > 0:
            self.logs['future_lataccel_0'].append(future_plan.lataccel[0])
            self.logs['future_roll_0'].append(future_plan.roll_lataccel[0])
            self.logs['future_v_ego_0'].append(future_plan.v_ego[0])
            self.logs['future_a_ego_0'].append(future_plan.a_ego[0])
            
            # Also log if we have 10 and 50 elements available
            if len(future_plan.lataccel) >= 10:
                self.logs['future_lataccel_10'].append(future_plan.lataccel[9])
            
            if len(future_plan.lataccel) >= 50:
                self.logs['future_lataccel_50'].append(future_plan.lataccel[49])
        
        # Standard PID logic
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        self.logs['action'].append(action)
        
        return action
    
    def print_diagnostics(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("CONTROLLER INPUT DIAGNOSTICS")
        print("="*60)
        print(f"\nTotal steps: {self.step}")
        
        for key in sorted(self.logs.keys()):
            data = np.array(self.logs[key])
            print(f"\n{key}:")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Min:  {np.min(data):.4f}")
            print(f"  Max:  {np.max(data):.4f}")
            
            # Show first few values
            if len(data) <= 5:
                print(f"  Values: {data}")
            else:
                print(f"  First 5: {data[:5]}")
        
        # Check future_plan length distribution
        plan_lens = np.array(self.logs['future_plan_len'])
        print(f"\nFuture plan length distribution:")
        unique, counts = np.unique(plan_lens, return_counts=True)
        for length, count in zip(unique, counts):
            print(f"  {int(length)} steps: {count} times ({count/len(plan_lens)*100:.1f}%)")
