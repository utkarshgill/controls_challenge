"""
Load the collected training data and verify it matches PID formula
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

def verify_pid_linearity():
    """Check if collected state-action pairs satisfy PID formula"""
    
    # Collect a small amount of data
    from train_v2_stateful import collect_pid_demonstrations_stateful, Path
    
    data_dir = Path('./data')
    data_files = sorted([str(f) for f in data_dir.glob('*.csv')])
    train_files = data_files[:15000]
    
    print("Collecting 10 files for verification...")
    states, actions = collect_pid_demonstrations_stateful(train_files, num_files=10)
    
    print(f"\n{'='*80}")
    print(f"Verification: Do state-action pairs satisfy PID formula?")
    print(f"{'='*80}\n")
    
    # PID coefficients
    P = 0.195
    I = 0.100
    D = -0.053
    
    # Check first 20 samples
    print(f"{'Idx':<6}{'Error':<12}{'ErrInt':<12}{'ErrDiff':<12}{'Action':<12}{'P*e+I*ei+D*ed':<15}{'Diff':<12}")
    print("-"*90)
    
    mismatches = 0
    for i in range(min(20, len(states))):
        error, error_int, error_diff = states[i]
        action = actions[i][0]
        
        predicted = P * error + I * error_int + D * error_diff
        diff = abs(action - predicted)
        
        match = "✓" if diff < 1e-5 else "✗"
        
        print(f"{i:<6}{error:<12.6f}{error_int:<12.6f}{error_diff:<12.6f}{action:<12.6f}{predicted:<15.6f}{diff:<12.8f} {match}")
        
        if diff >= 1e-5:
            mismatches += 1
    
    print(f"\n{'='*80}")
    if mismatches == 0:
        print(f"✅ ALL samples match PID formula perfectly!")
        print(f"   → Data collection is CORRECT")
        print(f"   → Problem must be elsewhere (normalization? optimization?)")
    else:
        print(f"❌ {mismatches} mismatches found!")
        print(f"   → Data collection has a BUG")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    verify_pid_linearity()



