"""
Evaluate BC controller using the OFFICIAL tinyphysics.py script
This matches the exact evaluation process described in the challenge README
"""
import sys
sys.path.insert(0, '../..')

import subprocess
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("Evaluating BC Controller (exp013) using OFFICIAL tinyphysics.py")
    print("="*80)
    print("\nThis evaluation uses the exact same process as the official challenge:")
    print("  python tinyphysics.py --controller <name> --data_path <path> --num_segs <n>")
    print()
    
    # Paths (resolve to absolute paths)
    root = Path(__file__).parent.parent.parent  # Go up to repo root
    model_path = root / "models/tinyphysics.onnx"
    data_path = root / "data"
    
    # Evaluation parameters
    # Use 100 routes for quick test (official submission uses 5000)
    num_segs = 100
    
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Routes: {num_segs}")
    print(f"  Controllers: PID (baseline), bc_exp013 (test)")
    print()
    
    # Check BC checkpoint exists
    checkpoint = Path("bc_best.pth")
    if not checkpoint.exists():
        print("❌ ERROR: bc_best.pth not found!")
        print("   Please train the BC network first: python train_bc.py")
        return 1
    
    print("✓ BC checkpoint found")
    print()
    
    # Evaluate PID (baseline)
    print("="*80)
    print("Evaluating PID (baseline)...")
    print("="*80)
    cmd_pid = [
        sys.executable,
        str(root / "tinyphysics.py"),
        "--model_path", str(model_path),
        "--data_path", str(data_path),
        "--num_segs", str(num_segs),
        "--controller", "pid"
    ]
    
    result_pid = subprocess.run(cmd_pid, cwd=str(root.resolve()), capture_output=True, text=True)
    print(result_pid.stdout)
    if result_pid.stderr:
        print("STDERR:", result_pid.stderr)
    
    # Evaluate BC
    print()
    print("="*80)
    print("Evaluating BC (exp013)...")
    print("="*80)
    cmd_bc = [
        sys.executable,
        str(root / "tinyphysics.py"),
        "--model_path", str(model_path),
        "--data_path", str(data_path),
        "--num_segs", str(num_segs),
        "--controller", "bc_exp013"
    ]
    
    result_bc = subprocess.run(cmd_bc, cwd=str(root.resolve()), capture_output=True, text=True)
    print(result_bc.stdout)
    if result_bc.stderr:
        print("STDERR:", result_bc.stderr)
    
    print()
    print("="*80)
    print("Evaluation complete!")
    print("="*80)
    print()
    print("To run the official eval.py comparison:")
    print(f"  cd {root}")
    print(f"  python eval.py --model_path {model_path} --data_path {data_path} \\")
    print(f"    --num_segs {num_segs} --test_controller bc_exp013 --baseline_controller pid")
    print()
    print("For official submission (5000 routes):")
    print(f"  python eval.py --model_path {model_path} --data_path {data_path} \\")
    print(f"    --num_segs 5000 --test_controller bc_exp013 --baseline_controller pid")
    print()


if __name__ == '__main__':
    main()
