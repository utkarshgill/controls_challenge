# The Simple Truth

**Goal**: Single linear layer, no bias, behavioral cloning from PID.

**What works**:
- Input: Just the error (or minimal features)
- Network: `steering = W @ features` (no bias!)
- Training: MSE loss on PID demonstrations
- That's it.

**Stop overcomplicating**:
- ❌ Stateful tracking (error_integral, prev_error)
- ❌ Trying to recover exact PID coefficients
- ❌ Analytical solutions
- ❌ Complex state representations

**Just**: Learn weights that minimize error on PID actions. Done.



