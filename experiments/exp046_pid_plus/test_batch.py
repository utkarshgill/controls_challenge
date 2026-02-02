"""Proper batch evaluation using official method"""

import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

if __name__ == '__main__':
    print("="*60)
    print("EXP046: Batch Evaluation")
    print("="*60)
    print("\nRunning official batch eval (100 routes)...")
    print("This is the CORRECT way to measure performance.\n")
    
    # Run official batch eval
    # For now, just test PID baseline
    # Later we'll add our controller to the controllers/ directory
    
    cmd = [
        "python", "tinyphysics.py",
        "--model_path", "./models/tinyphysics.onnx",
        "--data_path", "./data",
        "--num_segs", "100",
        "--controller", "pid"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent.parent,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("\n" + "="*60)
    print("BASELINE ESTABLISHED")
    print("="*60)
    print("\nPID batch average: ~84.85")
    print("Goal: Beat this with targeted improvements")
