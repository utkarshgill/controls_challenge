"""
Evaluate trained BC model against PID baseline.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import pid
from bc_controller import BCController


def evaluate_controller(controller, model, data_files, desc="Evaluating"):
    """
    Evaluate controller on multiple data files.
    
    Returns:
        costs: list of dicts with keys ['lataccel_cost', 'jerk_cost', 'total_cost']
    """
    costs = []
    
    for file_path in tqdm(data_files, desc=desc):
        # Reset controller state
        if hasattr(controller, 'reset'):
            controller.reset()
        
        # Run simulator
        sim = TinyPhysicsSimulator(model, str(file_path), controller=controller, debug=False)
        cost = sim.rollout()
        costs.append(cost)
    
    return costs


def main(model_path, data_dir, bc_checkpoint, num_files=100):
    """
    Compare BC controller to PID baseline.
    """
    
    # Load TinyPhysics model
    print(f"Loading TinyPhysics model from {model_path}...")
    tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
    
    # Get test files
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob("*.csv")))
    test_files = all_files[:num_files]
    
    print(f"\nEvaluating on {len(test_files)} files...")
    
    # Evaluate PID baseline
    print(f"\n{'='*70}")
    print("Evaluating PID baseline...")
    print(f"{'='*70}")
    pid_controller = pid.Controller()
    pid_costs = evaluate_controller(pid_controller, tinyphysics_model, test_files, desc="PID")
    
    # Evaluate BC controller
    print(f"\n{'='*70}")
    print(f"Evaluating BC controller from {bc_checkpoint}...")
    print(f"{'='*70}")
    bc_controller = BCController(bc_checkpoint)
    bc_costs = evaluate_controller(bc_controller, tinyphysics_model, test_files, desc="BC")
    
    # Compute statistics
    def compute_stats(costs):
        lataccel = np.array([c['lataccel_cost'] for c in costs])
        jerk = np.array([c['jerk_cost'] for c in costs])
        total = np.array([c['total_cost'] for c in costs])
        return {
            'lataccel': {'mean': lataccel.mean(), 'std': lataccel.std()},
            'jerk': {'mean': jerk.mean(), 'std': jerk.std()},
            'total': {'mean': total.mean(), 'std': total.std()}
        }
    
    pid_stats = compute_stats(pid_costs)
    bc_stats = compute_stats(bc_costs)
    
    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'PID':<20} {'BC':<20} {'Diff':<15}")
    print(f"{'-'*75}")
    
    print(f"{'Lataccel Cost':<20} "
          f"{pid_stats['lataccel']['mean']:.2f} ± {pid_stats['lataccel']['std']:.2f}  "
          f"{bc_stats['lataccel']['mean']:.2f} ± {bc_stats['lataccel']['std']:.2f}  "
          f"{bc_stats['lataccel']['mean'] - pid_stats['lataccel']['mean']:+.2f}")
    
    print(f"{'Jerk Cost':<20} "
          f"{pid_stats['jerk']['mean']:.2f} ± {pid_stats['jerk']['std']:.2f}  "
          f"{bc_stats['jerk']['mean']:.2f} ± {bc_stats['jerk']['std']:.2f}  "
          f"{bc_stats['jerk']['mean'] - pid_stats['jerk']['mean']:+.2f}")
    
    print(f"{'Total Cost':<20} "
          f"{pid_stats['total']['mean']:.2f} ± {pid_stats['total']['std']:.2f}  "
          f"{bc_stats['total']['mean']:.2f} ± {bc_stats['total']['std']:.2f}  "
          f"{bc_stats['total']['mean'] - pid_stats['total']['mean']:+.2f}")
    
    print(f"\n{'='*70}")
    
    # Determine if BC is competitive
    ratio = bc_stats['total']['mean'] / pid_stats['total']['mean']
    if ratio < 1.2:
        print(f"✓ BC is competitive with PID (within 20%)")
    elif ratio < 2.0:
        print(f"⚠ BC is acceptable (within 2x of PID)")
    else:
        print(f"✗ BC needs improvement ({ratio:.1f}x worse than PID)")
    
    print(f"\nRatio: {ratio:.2f}x")
    print(f"{'='*70}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BC model')
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx',
                        help='Path to TinyPhysics model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing CSV data files')
    parser.add_argument('--bc_checkpoint', type=str, 
                        default='./experiments/exp042/outputs/best_model.pt',
                        help='Path to trained BC model checkpoint')
    parser.add_argument('--num_files', type=int, default=100,
                        help='Number of files to evaluate on')
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        bc_checkpoint=args.bc_checkpoint,
        num_files=args.num_files
    )
