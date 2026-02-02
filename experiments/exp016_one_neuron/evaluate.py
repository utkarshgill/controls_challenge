"""
Evaluate trained model on test set using official tinyphysics evaluation

This uses the EXACT same evaluation as the official leaderboard.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
from data_split import get_data_split, print_split_info

def evaluate_controller(controller_name, num_segs=2500, split='test'):
    """
    Evaluate controller using official tinyphysics.py
    
    Args:
        controller_name: Name of controller (e.g., 'exp016_one_neuron')
        num_segs: Number of segments to evaluate (default: 2500 = full test set)
        split: Which split to use ('train', 'val', 'test')
    """
    print("="*80)
    print(f"Evaluating Controller: {controller_name}")
    print("="*80)
    print(f"Split: {split}")
    print(f"Segments: {num_segs}")
    print("="*80)
    
    # Show split info
    data_split = get_data_split()
    print(f"\nUsing {split} set: {len(data_split[split]):,} files available")
    print(f"Evaluating on: {min(num_segs, len(data_split[split])):,} segments\n")
    
    # Run official evaluation
    project_root = Path(__file__).parent.parent.parent
    
    cmd = [
        'python', 'tinyphysics.py',
        '--model_path', './models/tinyphysics.onnx',
        '--data_path', './data',
        '--controller', controller_name,
        '--num_segs', str(num_segs)
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    print("="*80)
    
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=False,  # Show output in real-time
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ Evaluation failed with exit code {result.returncode}")
    else:
        print(f"\n✅ Evaluation complete!")
    
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained controller')
    parser.add_argument('--controller', type=str, default='exp016_one_neuron',
                       help='Controller name')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    parser.add_argument('--num_segs', type=int, default=2500,
                       help='Number of segments (default: 2500 = full test set)')
    
    args = parser.parse_args()
    
    # Show data split info
    print_split_info()
    
    # Run evaluation
    success = evaluate_controller(args.controller, args.num_segs, args.split)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())



