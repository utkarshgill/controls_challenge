"""Test runner for exp046"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controller import Controller

if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("EXP046: PID Plus")
    print("="*60)
    print("\nBaseline: Pure PID")
    print("Expected: ~103 cost\n")
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = Controller()
    
    print("Running...")
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    print(f"\n{'='*60}")
    print(f"BASELINE RESULT:")
    print(f"  Total: {cost['total_cost']:.2f}")
    print(f"    Lat: {cost['lataccel_cost']:.2f} × 50 = {cost['lataccel_cost']*50:.1f}")
    print(f"    Jerk: {cost['jerk_cost']:.2f}")
    print(f"  Time: {(t1-t0):.1f}s")
    print(f"{'='*60}\n")
    
    if cost['total_cost'] < 90:
        print("✓ Already <90!")
    elif cost['total_cost'] < 120:
        print(f"Gap to <90: {cost['total_cost']-90:.1f}")
        print("\nNext: Diagnose where PID loses points")
    else:
        print("⚠ Worse than expected")
