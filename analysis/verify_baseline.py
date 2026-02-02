"""
Rigorous baseline verification.
Evaluate ALL our BC models on the EXACT SAME test set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random, numpy as np
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

model_onnx = TinyPhysicsModel('./models/tinyphysics.onnx', debug=False)

# EXACT test set
all_files = sorted(Path('./data').glob('*.csv'))
random.seed(42)
random.shuffle(all_files)
test_files = all_files[17500:20000]

print("="*70)
print("BASELINE VERIFICATION - Same 20 test routes for all controllers")
print("="*70)

# Test each controller
controllers_to_test = [
    ('PID', 'controllers.pid', 'Controller'),
    ('exp017 (1-neuron)', 'controllers.exp017_baseline', 'Controller'),
    ('exp023 (Conv1D)', 'experiments.exp023_conv.controller', 'Controller'),
    ('exp025 (Conv+history)', 'experiments.exp025_with_history.controller', 'Controller'),
]

results = {}

for name, module_path, class_name in controllers_to_test:
    print(f"\n{name}:")
    print("-" * 70)
    
    # Check if model exists
    if 'exp023' in module_path:
        model_path = Path('experiments/exp023_conv/model.pth')
    elif 'exp025' in module_path:
        model_path = Path('experiments/exp025_with_history/model.pth')
    elif 'exp017' in module_path:
        model_path = Path('experiments/exp017_baseline/model.pth')
    else:
        model_path = None
    
    if model_path and not model_path.exists():
        print(f"  ❌ Model not found: {model_path}")
        continue
    
    try:
        # Import controller
        module = __import__(module_path, fromlist=[class_name])
        ControllerClass = getattr(module, class_name)
        
        costs = []
        for i, f in enumerate(test_files[:20]):
            ctrl = ControllerClass()
            sim = TinyPhysicsSimulator(model_onnx, str(f), controller=ctrl)
            cost = sim.rollout()['total_cost']
            costs.append(cost)
            if i < 5:  # Show first 5
                print(f"  Route {i+1}: {cost:.2f}")
        
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        print(f"  ...")
        print(f"  Mean: {mean_cost:.2f} ± {std_cost:.2f}")
        print(f"  Range: [{np.min(costs):.2f}, {np.max(costs):.2f}]")
        
        results[name] = {
            'mean': mean_cost,
            'std': std_cost,
            'costs': costs
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if results:
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for name, data in sorted_results:
        print(f"{name:25s}: {data['mean']:6.2f} ± {data['std']:5.2f}")
    
    print("\n" + "="*70)
    best_name, best_data = sorted_results[0]
    print(f"✅ BEST BASELINE: {best_name}")
    print(f"   Cost: {best_data['mean']:.2f}")
    print("="*70)
else:
    print("❌ No controllers successfully evaluated")

