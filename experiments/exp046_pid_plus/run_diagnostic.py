"""Run diagnostic controller to understand inputs"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controller_diagnostic import Controller

if __name__ == '__main__':
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    data_path = Path(__file__).parent.parent.parent / 'data' / '00000.csv'
    
    print("="*60)
    print("DIAGNOSTIC RUN")
    print("="*60)
    print("\nRunning PID with logging on single trajectory...")
    
    model = TinyPhysicsModel(str(model_path), debug=False)
    controller = Controller()
    
    t0 = time.time()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    t1 = time.time()
    
    print(f"\nCost: {cost['total_cost']:.2f}")
    print(f"Time: {(t1-t0):.1f}s")
    
    # Print diagnostics
    controller.print_diagnostics()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
