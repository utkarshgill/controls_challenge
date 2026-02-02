"""
Trace EXACT PID logic step-by-step to understand state capture
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from controllers.pid import Controller as PIDController

def trace_pid_step_by_step():
    """
    Manually trace what PID does internally
    """
    print("="*80)
    print("PID Internal Logic Trace")
    print("="*80)
    
    # Create controller
    controller = PIDController()
    print(f"\nInitial state:")
    print(f"  error_integral = {controller.error_integral}")
    print(f"  prev_error = {controller.prev_error}")
    print(f"  P = {controller.p}, I = {controller.i}, D = {controller.d}")
    
    # Simulate one step
    target = 1.0
    current = 0.5
    
    print(f"\n{'='*80}")
    print(f"Step 1: target={target}, current={current}")
    print(f"{'='*80}")
    
    # What PID computes INSIDE update():
    print(f"\nInside update() method:")
    print(f"  1. error = {target} - {current} = {target - current}")
    
    # BEFORE any state updates:
    print(f"\n  2. State BEFORE updates:")
    print(f"     self.error_integral = {controller.error_integral}")
    print(f"     self.prev_error = {controller.prev_error}")
    
    error = target - current
    print(f"\n  3. self.error_integral += error  → {controller.error_integral} += {error}")
    controller.error_integral += error
    print(f"     → self.error_integral = {controller.error_integral}")
    
    print(f"\n  4. error_diff = error - self.prev_error")
    error_diff = error - controller.prev_error
    print(f"              = {error} - {controller.prev_error} = {error_diff}")
    
    print(f"\n  5. self.prev_error = error  → {error}")
    controller.prev_error = error
    
    print(f"\n  6. return = P * error + I * error_integral + D * error_diff")
    action = controller.p * error + controller.i * controller.error_integral + controller.d * error_diff
    print(f"          = {controller.p} * {error} + {controller.i} * {controller.error_integral} + {controller.d} * {error_diff}")
    print(f"          = {controller.p * error} + {controller.i * controller.error_integral} + {controller.d * error_diff}")
    print(f"          = {action}")
    
    print(f"\n{'='*80}")
    print(f"State used in formula:")
    print(f"  error: {error}")
    print(f"  error_integral: {controller.error_integral} (AFTER += error)")
    print(f"  error_diff: {error_diff} (using OLD prev_error)")
    print(f"={'='*80}\n")

if __name__ == '__main__':
    trace_pid_step_by_step()



