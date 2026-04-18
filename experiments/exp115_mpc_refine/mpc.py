#!/usr/bin/env python3
"""Receding-horizon MPC with learned prior.

At each control step:
  1. Policy proposes nominal delta-steer (Beta mean)
  2. Sample K perturbations around it
  3. Forward-simulate H steps for each candidate
  4. Pick best candidate's first action
  5. Execute on real sim, slide forward

Usage:
  .venv/bin/python experiments/exp115_mpc_refine/mpc.py --csv data/00000.csv
  .venv/bin/python experiments/exp115_mpc_refine/mpc.py --csv data/00000.csv --checkpoint path/to/best_model.pt
"""

import argparse, numpy as np, sys, time
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, ".")
from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER,
    LATACCEL_RANGE, VOCAB_SIZE, MAX_ACC_DELTA,
)
from controllers import BaseController

# MPC hyperparameters
H = 10           # planning horizon (steps to simulate forward)
K = 64           # number of candidate perturbations
SIGMA = 0.1      # perturbation std on raw delta [-1, 1]
DELTA_SCALE = 0.25


def simulate_horizon(phy_model, actions_H, context):
    """
    TODO(human): Simulate H steps forward and return the cost.

    Args:
        phy_model: TinyPhysicsModel (the ONNX physics model)
        actions_H: np.ndarray (H,) — absolute steer values for next H steps
        context: dict with keys:
            'action_history': list of last CL=20 steer values
            'state_history': list of last CL=20 State tuples (roll, v, a)
            'current_lataccel_history': list of last CL=20 lataccel predictions
            'current_lataccel': float — latest lataccel value
            'target_lataccel': np.ndarray (H,) — target lataccel for next H steps
            'states_ahead': list of H State tuples (road features for next H steps)

    Returns:
        float — average cost over the H-step horizon
               cost_per_step = lataccel_error² * LAT_ACCEL_COST_MULTIPLIER + jerk²
               return mean(cost_per_step) * 100
    """
    pass


class MPCController(BaseController):
    """Receding-horizon MPC using a trained policy as prior."""

    def __init__(self, checkpoint_path=None):
        self.phy_model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)

        # Load trained policy as prior (optional)
        self.prior = None
        if checkpoint_path and Path(checkpoint_path).exists():
            import torch
            from experiments.exp115_mpc_refine.train import ActorCritic, OBS_DIM
            ac = ActorCritic()
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            ac.load_state_dict(ckpt["ac"])
            ac.eval()
            self.prior = ac
            self._torch = torch
            self._F = torch.nn.functional

        self.step_count = 0
        self.h_act = [0.0] * CONTEXT_LENGTH
        self.h_lat = [0.0] * CONTEXT_LENGTH
        self.prev_steer = 0.0

    def _get_prior_delta(self, target, current, state, future_plan):
        """Get the trained policy's proposed delta (deterministic)."""
        if self.prior is None:
            return 0.0
        from experiments.exp115_mpc_refine.train import _build_obs_bc
        obs = _build_obs_bc(target, current, state, future_plan,
                            self.h_act, self.h_lat)
        with self._torch.no_grad():
            obs_t = self._torch.FloatTensor(obs).unsqueeze(0)
            a_p, b_p = self.prior.beta_params(obs_t)
            raw = 2.0 * a_p / (a_p + b_p) - 1.0
        return float(raw.item())

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1

        if self.step_count <= (CONTROL_START_IDX - CONTEXT_LENGTH):
            self.h_act = self.h_act[1:] + [0.0]
            self.h_lat = self.h_lat[1:] + [current_lataccel]
            return 0.0

        # Get prior's proposed delta
        prior_delta = self._get_prior_delta(
            target_lataccel, current_lataccel, state, future_plan)

        # Build context for forward simulation
        context = {
            'action_history': list(self.h_act),
            'state_history': [],  # filled from sim history
            'current_lataccel_history': list(self.h_lat),
            'current_lataccel': current_lataccel,
            'target_lataccel': np.array(
                getattr(future_plan, 'lataccel', [target_lataccel])[:H],
                dtype=np.float64),
        }

        # Sample K candidate deltas around the prior
        candidates = np.random.randn(K) * SIGMA + prior_delta
        candidates[0] = prior_delta  # elitism: include the prior's proposal
        candidates = np.clip(candidates, -1.0, 1.0)

        # Convert to absolute steer sequences
        best_cost = float('inf')
        best_action = self.prev_steer + prior_delta * DELTA_SCALE

        for k in range(K):
            # First action from this candidate
            first_delta = candidates[k] * DELTA_SCALE
            first_action = np.clip(self.prev_steer + first_delta,
                                   STEER_RANGE[0], STEER_RANGE[1])

            # For the horizon, use the prior's delta repeated (simple)
            actions_H = np.full(H, first_action)
            for h in range(1, H):
                actions_H[h] = np.clip(actions_H[h-1] + prior_delta * DELTA_SCALE,
                                       STEER_RANGE[0], STEER_RANGE[1])

            cost = simulate_horizon(self.phy_model, actions_H, context)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_action = first_action

        action = float(np.clip(best_action, STEER_RANGE[0], STEER_RANGE[1]))
        self.h_act = self.h_act[1:] + [action]
        self.h_lat = self.h_lat[1:] + [current_lataccel]
        self.prev_steer = action
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to route CSV")
    parser.add_argument("--checkpoint", default="", help="Path to trained .pt")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.1)
    args = parser.parse_args()

    global H, K, SIGMA
    H, K, SIGMA = args.horizon, args.samples, args.sigma

    model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    controller = MPCController(checkpoint_path=args.checkpoint or None)
    sim = TinyPhysicsSimulator(model, args.csv, controller, debug=False)

    t0 = time.time()
    cost = sim.rollout()
    dt = time.time() - t0

    print(f"Route: {args.csv}")
    print(f"Cost: {cost['total_cost']:.2f}  (lat={cost['lataccel_cost']:.2f} jerk={cost['jerk_cost']:.2f})")
    print(f"Time: {dt:.1f}s")

    # Save actions
    actions = np.array(sim.action_history[CONTROL_START_IDX:COST_END_IDX])
    out = Path("experiments/exp115_mpc_refine") / f"actions_{Path(args.csv).stem}.npz"
    np.savez(out, actions=actions)
    print(f"Saved {len(actions)} actions to {out}")


if __name__ == "__main__":
    main()
