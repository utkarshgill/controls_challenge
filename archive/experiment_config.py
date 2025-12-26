#!/usr/bin/env python3
"""
Experiment Configuration System

Tracks different state designs and their performance.
"""

import numpy as np
import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_dir="experiments"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(self, name, config):
        """Start a new experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp}_{name}"
        exp_path = os.path.join(self.experiment_dir, exp_id)
        os.makedirs(exp_path, exist_ok=True)
        
        self.current_experiment = {
            'id': exp_id,
            'name': name,
            'path': exp_path,
            'config': config,
            'start_time': timestamp,
            'results': {}
        }
        
        # Save config
        with open(os.path.join(exp_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"ID: {exp_id}")
        print(f"{'='*60}")
        print(f"Config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
        
        return exp_id
    
    def log_results(self, results):
        """Log experiment results"""
        if self.current_experiment is None:
            raise ValueError("No experiment started")
        
        self.current_experiment['results'] = results
        
        # Save results
        results_path = os.path.join(self.current_experiment['path'], 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_path = os.path.join(self.current_experiment['path'], 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {self.current_experiment['name']}\n")
            f.write(f"ID: {self.current_experiment['id']}\n")
            f.write(f"Start: {self.current_experiment['start_time']}\n\n")
            f.write("Config:\n")
            for k, v in self.current_experiment['config'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\nResults:\n")
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
        
        print(f"\n✅ Results saved to {self.current_experiment['path']}")
        
        return results_path

# State design configurations
STATE_DESIGNS = {
    'current': {
        'name': 'Current (no a_ego)',
        'features': ['error', 'error_diff', 'error_integral', 'current_lataccel', 'v_ego', 'curv_now'] + ['future_curv'] * 50,
        'dim': 56,
        'includes_a_ego': False,
        'obs_scale': [10.0, 1.0, 0.1, 2.0, 0.03, 1000.0] + [1000.0] * 50,
    },
    'with_a_ego': {
        'name': 'With a_ego (friction circle)',
        'features': ['error', 'error_diff', 'error_integral', 'current_lataccel', 'v_ego', 'a_ego', 'curv_now'] + ['future_curv'] * 50,
        'dim': 57,
        'includes_a_ego': True,
        'obs_scale': [10.0, 1.0, 0.1, 2.0, 0.03, 20.0, 1000.0] + [1000.0] * 50,
    },
    'friction_aware': {
        'name': 'Friction circle aware',
        'features': ['error', 'error_diff', 'error_integral', 'current_lataccel', 'v_ego', 'a_ego', 
                    'friction_margin', 'curv_now'] + ['future_curv'] * 50,
        'dim': 58,
        'includes_a_ego': True,
        'includes_friction_margin': True,
        'obs_scale': [10.0, 1.0, 0.1, 2.0, 0.03, 20.0, 1.0, 1000.0] + [1000.0] * 50,
    }
}

def compute_friction_margin(a_lat, a_long, mu_g=9.8):
    """
    Compute available friction margin
    
    Friction circle: sqrt(a_lat² + a_long²) ≤ μ·g
    Margin = how much of the friction circle is left
    
    Returns: [0, 1] where 1 = no acceleration, 0 = at limit
    """
    total_accel = np.sqrt(a_lat**2 + a_long**2)
    margin = max(0, 1 - total_accel / mu_g)
    return margin

def print_state_comparison():
    """Print comparison of different state designs"""
    print("\n" + "="*80)
    print("STATE DESIGN OPTIONS")
    print("="*80)
    
    for key, design in STATE_DESIGNS.items():
        print(f"\n{key.upper()}:")
        print(f"  Name: {design['name']}")
        print(f"  Dim: {design['dim']}")
        print(f"  Features: {', '.join(design['features'][:8])} + {len(design['features'])-8} more")
        print(f"  Includes a_ego: {design.get('includes_a_ego', False)}")
        print(f"  Includes friction margin: {design.get('includes_friction_margin', False)}")
    
    print("\n" + "="*80)
    print("HYPOTHESIS: Including a_ego")
    print("="*80)
    print("""
When braking/accelerating hard (high |a_ego|):
  → Less friction available for lateral control
  → Same steering input has different effect
  → Network needs to know a_ego to predict correctly

File 00069 has lots of speed changes (v ∈ [0, 5])
  → High |a_ego| throughout
  → BC without a_ego can't model this coupling
  → PPO could learn it from trial & error, but needs more training
    """)
    print("="*80)

if __name__ == '__main__':
    print_state_comparison()

