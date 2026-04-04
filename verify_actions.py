"""Quick verification: replay actions on official CPU sim."""

import numpy as np, sys

sys.path.insert(0, ".")
from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    CONTROL_START_IDX,
    CONTEXT_LENGTH,
)


class LookupCtrl:
    def __init__(self, acts):
        self.acts = acts
        self.step = 0

    def update(self, target, current, state, future_plan=None):
        self.step += 1
        # step=1 is first call at step_idx=CONTEXT_LENGTH(20)
        # MPC actions start at step_idx=CONTROL_START_IDX(100) → step=81
        ci = self.step - (CONTROL_START_IDX - CONTEXT_LENGTH + 1)
        if 0 <= ci < len(self.acts):
            return float(self.acts[ci])
        return 0.0


NPZ = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "experiments/exp110_mpc/checkpoints/actions.npz"
)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

actions = np.load(NPZ)
costs = []
for name in sorted(actions.files)[:N]:
    acts = actions[name]
    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    sim = TinyPhysicsSimulator(model, f"data/{name}", LookupCtrl(acts))
    cost = sim.rollout()
    tc = cost["total_cost"]
    costs.append(tc)
    print(f"  {name}: {tc:.2f}")

print(f"Mean ({len(costs)} routes): {np.mean(costs):.2f}")
