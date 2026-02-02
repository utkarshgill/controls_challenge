"""Test PID baseline on the 400 eval routes"""

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
    eval_routes = sorted(list(data_dir.glob('*.csv')))[2000:2400]  # 400 eval routes
    
    print(f'Testing PID on {len(eval_routes)} eval routes...')
    test_partial = partial(test_route, model_path=model_path)
    costs = process_map(test_partial, eval_routes, max_workers=16, chunksize=10)
    
    print(f'\nPID on 400 eval routes:')
    print(f'  Mean: {np.mean(costs):.1f}')
    print(f'  Median: {np.median(costs):.1f}')
    print(f'  Std: {np.std(costs):.1f}')
    print(f'\nFor comparison:')
    print(f'  Batch metrics (100 random): 84.85 mean')
    print(f'  PPO eval reports: 71.0 median')
