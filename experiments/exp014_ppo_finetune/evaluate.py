"""
Evaluate PPO controller (exp014) using official tinyphysics.py
"""
import sys
sys.path.insert(0, '../..')

import subprocess
from pathlib import Path

def main():
    print("="*80)
    print("Evaluating PPO Controller (exp014) using OFFICIAL tinyphysics.py")
    print("="*80)
    print()
    print("This evaluation uses the exact same process as the official challenge:")
    print("  python tinyphysics.py --controller <name> --data_path <path> --num_segs <n>")
    print()
    
    # Paths
    root_dir = Path(__file__).parent.parent.parent
    model_path = root_dir / "models/tinyphysics.onnx"
    data_path = root_dir / "data"
    
    # Check if checkpoint exists
    checkpoint_path = Path(__file__).parent / "ppo_best.pth"
    if not checkpoint_path.exists():
        print(f"❌ PPO checkpoint not found: {checkpoint_path}")
        print("   Please train first: python train_ppo.py")
        return
    print("✓ PPO checkpoint found")
    print()
    
    # Configuration
    num_routes = 100
    controllers = [
        ('pid', 'PID (baseline)'),
        ('bc_exp013', 'BC (exp013)'),
        ('ppo_exp014', 'PPO (exp014)')
    ]
    
    print("Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Routes: {num_routes}")
    print(f"  Controllers: {', '.join([name for name, _ in controllers])}")
    print()
    
    results = {}
    
    for controller_name, display_name in controllers:
        print("="*80)
        print(f"Evaluating {display_name}...")
        print("="*80)
        print()
        
        # Run evaluation
        cmd = [
            'python', 'tinyphysics.py',
            '--model_path', str(model_path),
            '--data_path', str(data_path),
            '--num_segs', str(num_routes),
            '--controller', controller_name
        ]
        
        result = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True)
        
        # Parse output
        if result.returncode == 0:
            # Extract cost from output
            for line in result.stdout.split('\n'):
                if 'Average' in line and 'cost' in line:
                    print(line)
                    results[controller_name] = line
        else:
            print(f"❌ Error running {controller_name}")
            print(result.stderr)
        
        print()
        print(f"STDERR: \n{result.stderr}\n")
        print()
    
    print("="*80)
    print("Evaluation complete!")
    print("="*80)
    print()
    
    # Print comparison
    if len(results) > 1:
        print("Comparison:")
        for controller_name, output in results.items():
            print(f"  {controller_name}: {output}")
        print()
    
    print("To run the official eval.py comparison:")
    print(f"  cd {root_dir}")
    print(f"  python eval.py --model_path {model_path} --data_path {data_path} \\")
    print(f"    --num_segs {num_routes} --test_controller ppo_exp014 --baseline_controller pid")
    print()
    print("For official submission (5000 routes):")
    print(f"  python eval.py --model_path {model_path} --data_path {data_path} \\")
    print(f"    --num_segs 5000 --test_controller ppo_exp014 --baseline_controller pid")
    print()


if __name__ == '__main__':
    main()



