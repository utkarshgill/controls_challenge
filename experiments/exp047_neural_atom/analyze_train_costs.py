"""Analyze train vs eval cost distributions"""

import sys
from pathlib import Path
import numpy as np
from tqdm.contrib.concurrent import process_map
from functools import partial

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers.pid import Controller

def test_route(route, model_path):
    model = TinyPhysicsModel(model_path, debug=False)
    controller = Controller()
    sim = TinyPhysicsSimulator(model, str(route), controller=controller, debug=False)
    sim.rollout()
    return sim.compute_cost()['total_cost']

if __name__ == '__main__':
    model_path = './models/tinyphysics.onnx'
    data_dir = Path('./data')
    
    # Get train and eval routes
    all_routes = sorted(list(data_dir.glob('*.csv')))
    train_routes = all_routes[:2000]
    eval_routes = all_routes[2000:2400]
    
    print("Testing PID baseline on train vs eval sets...")
    print(f"Train: {len(train_routes)} routes")
    print(f"Eval: {len(eval_routes)} routes")
    
    test_partial = partial(test_route, model_path=model_path)
    
    # Sample 200 from each
    train_sample = np.random.choice(train_routes, 200, replace=False)
    eval_sample = eval_routes
    
    print(f"\nRunning on {len(train_sample)} train routes...")
    train_costs = process_map(test_partial, train_sample, max_workers=16, chunksize=10)
    
    print(f"Running on {len(eval_sample)} eval routes...")
    eval_costs = process_map(test_partial, eval_sample, max_workers=16, chunksize=10)
    
    print(f"\n{'='*60}")
    print("PID BASELINE COMPARISON:")
    print(f"{'='*60}")
    print(f"Train routes (n={len(train_costs)}):")
    print(f"  Mean:   {np.mean(train_costs):.1f}")
    print(f"  Median: {np.median(train_costs):.1f}")
    print(f"  Std:    {np.std(train_costs):.1f}")
    
    print(f"\nEval routes (n={len(eval_costs)}):")
    print(f"  Mean:   {np.mean(eval_costs):.1f}")
    print(f"  Median: {np.median(eval_costs):.1f}")
    print(f"  Std:    {np.std(eval_costs):.1f}")
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"{'='*60}")
    print(f"If train uses MEAN and eval uses MEDIAN:")
    print(f"  Train would report: ~{np.mean(train_costs):.1f}")
    print(f"  Eval would report:  ~{np.median(eval_costs):.1f}")
    print(f"  Difference: {np.mean(train_costs) - np.median(eval_costs):.1f}")
    print(f"\nBut if both use MEDIAN:")
    print(f"  Train: {np.median(train_costs):.1f}")
    print(f"  Eval:  {np.median(eval_costs):.1f}")
    print(f"  Difference: {np.median(train_costs) - np.median(eval_costs):.1f}")
