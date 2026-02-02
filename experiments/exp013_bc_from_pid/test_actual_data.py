"""
Test what cost the actual data steering commands achieve
"""
import sys
sys.path.insert(0, '../..')

from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from pathlib import Path
import numpy as np

class DataController:
    """Controller that uses the actual steering commands from the CSV"""
    def __init__(self):
        pass
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # This will never be called because we set actions directly
        return 0.0


model_path = "../../models/tinyphysics.onnx"
model = TinyPhysicsModel(model_path, debug=False)

# Test on 10 files
data_dir = Path("../../data")
test_files = list(data_dir.glob("*.csv"))[:10]

costs = []
for data_file in test_files:
    controller = DataController()
    sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
    
    # Use actual steering commands from CSV
    # The simulator already loads these into sim.data['steer_command']
    # By default, actions before CONTROL_START_IDX use the data commands
    # Let's see what happens if we use data commands for everything
    
    # Run simulation - it will use data commands by default
    for _ in range(len(sim.data) - 50):
        sim.step()
    
    cost = sim.compute_cost()
    costs.append(cost['total_cost'])
    print(f"{data_file.name}: {cost['total_cost']:.2f}")

print(f"\nActual data steering commands:")
print(f"  Mean: {np.mean(costs):.2f}")
print(f"  Std: {np.std(costs):.2f}")
print(f"  Min: {np.min(costs):.2f}")
print(f"  Max: {np.max(costs):.2f}")

