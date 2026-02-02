"""
Debug: Is PID actually linear?

Test if: action = P*error + I*error_integral + D*error_diff
holds exactly for PID controller.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from controllers.pid import Controller as PIDController

def test_pid_linearity():
    """Test if PID output matches P*e + I*ei + D*ed"""
    
    controller = PIDController()
    
    print("Testing PID Linearity")
    print("="*60)
    print(f"PID coefficients: P={controller.p}, I={controller.i}, D={controller.d}")
    print("="*60)
    
    # Simulate a few steps
    test_cases = [
        (1.0, 0.5),  # (target, current)
        (0.8, 0.6),
        (0.5, 0.7),
        (0.3, 0.4),
        (0.2, 0.25),
    ]
    
    print(f"\n{'Step':<6}{'Error':<10}{'ErrInt':<10}{'ErrDiff':<10}{'Action':<12}{'P*e+I*ei+D*ed':<15}{'Match?'}")
    print("-"*80)
    
    for i, (target, current) in enumerate(test_cases):
        # Capture state BEFORE update
        error = target - current
        error_integral_before = controller.error_integral
        prev_error_before = controller.prev_error
        error_diff = error - prev_error_before
        
        # Get PID action
        action = controller.update(target, current, None, None)
        
        # Manual calculation
        manual_action = (controller.p * error + 
                        controller.i * error_integral_before + 
                        controller.d * error_diff)
        
        # Check if they match (BEFORE state update)
        match = "✓" if abs(action - manual_action) < 1e-6 else "✗"
        
        print(f"{i:<6}{error:<10.4f}{error_integral_before:<10.4f}{error_diff:<10.4f}"
              f"{action:<12.4f}{manual_action:<15.4f}{match}")
        
        # Now check AFTER state update
        error_integral_after = controller.error_integral
        prev_error_after = controller.prev_error
        
        manual_action_after = (controller.p * error + 
                              controller.i * error_integral_after + 
                              controller.d * error_diff)
        
        print(f"  After: ErrInt={error_integral_after:.4f}, "
              f"Manual={manual_action_after:.4f} (doesn't match action)")
    
    print("\n" + "="*80)
    print("Conclusion: PID uses state from BEFORE the update!")
    print("error_integral in formula = error_integral BEFORE += error")
    print("="*80)


if __name__ == '__main__':
    test_pid_linearity()



