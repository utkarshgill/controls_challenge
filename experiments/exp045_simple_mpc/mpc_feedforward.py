"""
Step 1: Pure Feedforward Baseline

No optimization, no CEM, no complexity.
Just: action = Kp * (target - current)

This is the SIMPLEST possible controller.
If this doesn't beat PID, nothing else will matter.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController, STEER_RANGE


class SimpleController(BaseController):
    """Exact PID from controllers/pid.py"""
    
    def __init__(self):
        super().__init__()
        self.p = 0.195
        self.i = 0.100
        self.d = -0.053
        self.error_integral = 0
        self.prev_error = 0
        
    def update(self, target, current, state, future_plan):
        error = target - current
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        action = self.p * error + self.i * self.error_integral + self.d * error_diff
        return float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))


if __name__ == '__main__':
    import time
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("BASELINE: Exact PID (sanity check)")
    print("="*60)
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = SimpleController()  # Exact PID
    
    print("\nRunning...")
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Cost: {cost['total_cost']:.2f}")
    print(f"    Lat: {cost['lataccel_cost']:.2f} × 50 = {cost['lataccel_cost']*50:.1f}")
    print(f"    Jerk: {cost['jerk_cost']:.2f}")
    print(f"  Time: {(t1-t0):.1f}s")
    print(f"{'='*60}")
    print(f"\nPID baseline: ~90-120")
    print(f"This: {cost['total_cost']:.1f}")
    
    if cost['total_cost'] < 120:
        print("✓ Better than PID!")
    else:
        print("✗ Worse than PID")
