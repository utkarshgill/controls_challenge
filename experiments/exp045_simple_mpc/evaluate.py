"""
Evaluate MPC controller on validation set.
"""

import numpy as np
from pathlib import Path
import sys
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from mpc_controller import SimpleMPC, SimpleMPC_WithCEM


def eval_worker(args):
    """Worker for parallel evaluation"""
    csv_path, controller_type, params = args
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'tinyphysics.onnx'
    model = TinyPhysicsModel(str(model_path), debug=False)
    
    # Create controller
    if controller_type == 'simple':
        controller = SimpleMPC(model, **params)
    elif controller_type == 'cem':
        controller = SimpleMPC_WithCEM(model, **params)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Run simulation
    sim = TinyPhysicsSimulator(model, str(csv_path), controller=controller, debug=False)
    cost_dict = sim.rollout()
    
    return cost_dict['total_cost']


def main():
    print("=== Exp045: Simple MPC Evaluation ===\n")
    
    # Load validation data
    data_dir = Path(__file__).parent.parent.parent / 'data'
    csv_files = sorted(data_dir.glob('*.csv'))
    
    # Use 100 files for quick eval
    eval_files = csv_files[5000:5100]
    
    print(f"Evaluating on {len(eval_files)} files\n")
    
    # Test 1: Simple MPC (random shooting)
    print("1. SimpleMPC (horizon=10, samples=1000)...")
    params = {'horizon': 10, 'num_samples': 1000, 'action_std': 0.3}
    args = [(f, 'simple', params) for f in eval_files]
    costs_simple = process_map(eval_worker, args, max_workers=8, chunksize=10)
    mean_simple = np.mean(costs_simple)
    print(f"   Mean cost: {mean_simple:.2f}\n")
    
    # Test 2: MPC with CEM
    print("2. SimpleMPC_WithCEM (horizon=10, samples=500, cem_iters=3)...")
    params_cem = {'horizon': 10, 'num_samples': 500, 'num_elites': 50, 'cem_iterations': 3, 'action_std': 0.3}
    args_cem = [(f, 'cem', params_cem) for f in eval_files]
    costs_cem = process_map(eval_worker, args_cem, max_workers=8, chunksize=10)
    mean_cem = np.mean(costs_cem)
    print(f"   Mean cost: {mean_cem:.2f}\n")
    
    # Test 3: Longer horizon
    print("3. SimpleMPC (horizon=20, samples=2000)...")
    params_long = {'horizon': 20, 'num_samples': 2000, 'action_std': 0.3}
    args_long = [(f, 'simple', params_long) for f in eval_files]
    costs_long = process_map(eval_worker, args_long, max_workers=8, chunksize=10)
    mean_long = np.mean(costs_long)
    print(f"   Mean cost: {mean_long:.2f}\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print(f"  SimpleMPC (H=10, N=1000):      {mean_simple:.2f}")
    print(f"  SimpleMPC_WithCEM (H=10, N=500): {mean_cem:.2f}")
    print(f"  SimpleMPC (H=20, N=2000):      {mean_long:.2f}")
    print(f"  PID Baseline:                   ~126")
    print("=" * 60)


if __name__ == '__main__':
    main()
