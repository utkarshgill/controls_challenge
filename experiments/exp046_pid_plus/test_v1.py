"""Quick test of v1 (single trajectory for speed)"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controller_v1_roll import Controller

if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("v1: PID + Roll Compensation")
    print("="*60)
    print("\nQuick test on single trajectory (00000.csv)")
    print("Baseline on this trajectory: 102.87\n")
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = Controller()
    
    print("Running...")
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Total: {cost['total_cost']:.2f}")
    print(f"    Lat: {cost['lataccel_cost']:.2f} × 50 = {cost['lataccel_cost']*50:.1f}")
    print(f"    Jerk: {cost['jerk_cost']:.2f}")
    print(f"  Time: {(t1-t0):.1f}s")
    print(f"{'='*60}\n")
    
    baseline = 102.87
    delta = cost['total_cost'] - baseline
    
    if delta < -1:
        print(f"✓ Improvement: {-delta:.2f} points better")
        print("\nNext: Deploy to controllers/ and run batch eval")
    elif delta < 1:
        print(f"~ No change: {delta:+.2f}")
        print("\nTry adjusting roll_gain parameter")
    else:
        print(f"✗ Worse: {delta:+.2f} points")
        print("\nRoll compensation might not help, or wrong sign/magnitude")
