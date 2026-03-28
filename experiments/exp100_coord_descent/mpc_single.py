# Single-route MPC. Dead simple.
#
# Load one route. Run step by step.
# At each control step:
#   1. Get current state from sim
#   2. Sample K candidate actions from policy ± noise
#   3. For each candidate: clone the sim state, run H steps forward
#   4. Compute cost against future plan
#   5. Pick the best first action, execute it
#
# Uses tinyphysics.py (CPU, single route) for simplicity.
# Once this works, we parallelize.

import numpy as np, os, sys, time
from pathlib import Path
from hashlib import md5
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    TinyPhysicsModel,
    TinyPhysicsSimulator,
    CONTROL_START_IDX,
    COST_END_IDX,
    CONTEXT_LENGTH,
    STEER_RANGE,
    DEL_T,
    LAT_ACCEL_COST_MULTIPLIER,
    MAX_ACC_DELTA,
)

MPC_K = int(os.getenv("MPC_K", "32"))
MPC_H = int(os.getenv("MPC_H", "20"))
MPC_SIGMA = float(os.getenv("MPC_SIGMA", "0.01"))


class MpcController:
    """MPC controller that does lookahead search at each step."""

    def __init__(self, model_path, data_path):
        self.model = TinyPhysicsModel(model_path, debug=False)
        self.data_path = data_path
        self.n = 0
        self.prev_action = 0.0

        # Load the PPO policy for warm start
        import torch, torch.nn as nn, torch.nn.functional as F

        base_pt = ROOT / "experiments" / "exp055_batch_of_batch" / "best_model.pt"
        # For simplicity, just use a fixed prior: previous action (zero-order hold)
        # The real version would load the policy
        self.use_policy = False

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.n += 1
        if self.n <= CONTROL_START_IDX - CONTEXT_LENGTH:
            return 0.0

        # Get future targets for cost computation
        future_targets = []
        if future_plan and hasattr(future_plan, "lataccel") and future_plan.lataccel:
            future_targets = list(future_plan.lataccel[:MPC_H])
        # Pad with current target if not enough
        while len(future_targets) < MPC_H:
            future_targets.append(target_lataccel)

        # Sample K candidate first actions
        base_action = self.prev_action
        candidates = [base_action]  # always include current
        for _ in range(MPC_K - 1):
            noise = np.random.randn() * MPC_SIGMA
            cand = np.clip(base_action + noise, STEER_RANGE[0], STEER_RANGE[1])
            candidates.append(cand)

        # Evaluate each candidate via H-step rollout
        best_cost = float("inf")
        best_action = base_action

        for cand in candidates:
            # Save sim state
            saved_rng = np.random.get_state()

            # Simulate H steps with this candidate
            # We need to clone the sim's internal state
            # For now, use a fresh mini-sim that approximates
            cost = self._evaluate_candidate(
                cand, current_lataccel, target_lataccel, future_targets, saved_rng
            )

            if cost < best_cost:
                best_cost = cost
                best_action = cand

        self.prev_action = best_action
        return best_action

    def _evaluate_candidate(
        self, action, current_la, target, future_targets, rng_state
    ):
        """Evaluate a candidate action by simulating H steps.

        This is simplified — it doesn't run the full physics model.
        It uses a simple approximation: the lataccel moves toward the action
        with some lag. For the real version, we'd clone the sim.
        """
        # Simple cost: how close is this action to tracking the future plan?
        # The action enters the physics model which predicts lataccel.
        # Without running the model, approximate: lataccel ~ prev + small_change
        la = current_la
        prev_la = la
        cost = 0.0
        for h in range(min(MPC_H, len(future_targets))):
            # Very rough: the physics moves lataccel toward where the action points
            # This is a terrible approximation but shows the structure
            tgt = future_targets[h]
            cost += (tgt - la) ** 2 * LAT_ACCEL_COST_MULTIPLIER
            if h > 0:
                cost += ((la - prev_la) / DEL_T) ** 2
            prev_la = la
        return cost


def run_single_route(csv_path):
    """Run MPC on a single route using the real tinyphysics sim."""
    model = TinyPhysicsModel(str(ROOT / "models" / "tinyphysics.onnx"), debug=False)

    # We need to run the sim step by step with MPC at each control step.
    # The sim's rollout() calls controller.update() at each step.
    # Inside update(), we do the MPC lookahead.
    #
    # The problem: MPC lookahead needs to run the physics model forward,
    # which means we need to clone the sim's state (histories, RNG).
    #
    # Let's do it properly: at each step, save the sim state,
    # run K candidates forward H steps each, restore, pick best.

    import pandas as pd
    from tinyphysics import ACC_G

    df = pd.read_csv(str(csv_path))
    data = {
        "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
        "v_ego": df["vEgo"].values,
        "a_ego": df["aEgo"].values,
        "target_lataccel": df["targetLateralAcceleration"].values,
        "steer_command": -df["steerCommand"].values,
    }
    n_steps = len(df)
    seed = int(md5(str(csv_path).encode()).hexdigest(), 16) % 10**4

    # Initialize sim state
    np.random.seed(seed)

    # Physics model for lookahead
    lookahead_model = TinyPhysicsModel(
        str(ROOT / "models" / "tinyphysics.onnx"), debug=False
    )

    # Main sim state
    state_history = []
    action_history = []
    lataccel_history = []

    for i in range(CONTEXT_LENGTH):
        state_history.append(
            (data["roll_lataccel"][i], data["v_ego"][i], data["a_ego"][i])
        )
        action_history.append(data["steer_command"][i])
        lataccel_history.append(data["target_lataccel"][i])

    current_lataccel = lataccel_history[-1]
    prev_action = 0.0

    # PPO policy warm start for base actions
    # For now just use 0 as base (will add policy later)

    all_actions = []

    for step_idx in range(CONTEXT_LENGTH, n_steps):
        row_state = (
            data["roll_lataccel"][step_idx],
            data["v_ego"][step_idx],
            data["a_ego"][step_idx],
        )
        target = data["target_lataccel"][step_idx]

        if step_idx < CONTROL_START_IDX:
            action = data["steer_command"][step_idx]
        else:
            # ── MPC: sample K candidates, evaluate H steps, pick best ──
            # Save RNG state
            saved_rng = np.random.get_state()

            # Future targets
            future_targets = data["target_lataccel"][
                step_idx : step_idx + MPC_H
            ].tolist()
            while len(future_targets) < MPC_H:
                future_targets.append(target)

            base_action = prev_action
            candidates = [base_action]
            for _ in range(MPC_K - 1):
                cand = np.clip(
                    base_action + np.random.randn() * MPC_SIGMA,
                    STEER_RANGE[0],
                    STEER_RANGE[1],
                )
                candidates.append(cand)

            best_cost = float("inf")
            best_action = base_action

            for cand in candidates:
                # Restore RNG to exact state before this step's sim_step
                np.random.set_state(saved_rng)

                # Clone histories
                tmp_state = list(state_history)
                tmp_action = list(action_history)
                tmp_lataccel = list(lataccel_history)
                tmp_current = current_lataccel

                # Simulate H steps
                cost = 0.0
                prev_la = tmp_current

                for h in range(MPC_H):
                    s = step_idx + h
                    if s >= n_steps:
                        break

                    act = cand if h == 0 else prev_action  # zero-order hold for h>0

                    # Physics step
                    s_state = (
                        data["roll_lataccel"][s],
                        data["v_ego"][s],
                        data["a_ego"][s],
                    )
                    tmp_state.append(s_state)
                    act = np.clip(act, STEER_RANGE[0], STEER_RANGE[1])
                    tmp_action.append(act)

                    CL = CONTEXT_LENGTH
                    sim_states = tmp_state[-CL:]
                    sim_actions = tmp_action[-CL:]
                    past_preds = tmp_lataccel[-CL:]

                    pred = lookahead_model.get_current_lataccel(
                        sim_states=sim_states,
                        actions=sim_actions,
                        past_preds=past_preds,
                    )
                    pred = np.clip(
                        pred, tmp_current - MAX_ACC_DELTA, tmp_current + MAX_ACC_DELTA
                    )

                    if s >= CONTROL_START_IDX:
                        tmp_current = pred
                    else:
                        tmp_current = data["target_lataccel"][s]

                    tmp_lataccel.append(tmp_current)

                    # Cost against future plan
                    if CONTROL_START_IDX <= s < COST_END_IDX:
                        tgt = data["target_lataccel"][s]
                        cost += (
                            (tgt - tmp_current) ** 2 * 100 * LAT_ACCEL_COST_MULTIPLIER
                        )
                        if h > 0:
                            cost += ((tmp_current - prev_la) / DEL_T) ** 2 * 100
                    prev_la = tmp_current

                if cost < best_cost:
                    best_cost = cost
                    best_action = cand

            action = best_action
            # Restore RNG for the REAL sim step
            np.random.set_state(saved_rng)

        # Execute the chosen action in the real sim
        state_history.append(row_state)
        action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
        action_history.append(action)

        # Real physics step (consumes one RNG draw)
        CL = CONTEXT_LENGTH
        pred = model.get_current_lataccel(
            sim_states=state_history[-CL:],
            actions=action_history[-CL:],
            past_preds=lataccel_history[-CL:],
        )
        pred = np.clip(
            pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA
        )

        if step_idx >= CONTROL_START_IDX:
            current_lataccel = pred
        else:
            current_lataccel = data["target_lataccel"][step_idx]

        lataccel_history.append(current_lataccel)
        prev_action = action

        if step_idx >= CONTROL_START_IDX and step_idx < COST_END_IDX:
            all_actions.append(action)

    # Compute final cost
    scored = lataccel_history[CONTROL_START_IDX:COST_END_IDX]
    targets = data["target_lataccel"][CONTROL_START_IDX:COST_END_IDX]
    lat_cost = np.mean((targets - scored) ** 2) * 100
    jerk = np.diff(scored) / DEL_T
    jerk_cost = np.mean(jerk**2) * 100
    total = lat_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

    return total, all_actions


if __name__ == "__main__":
    csv_path = sorted((ROOT / "data").glob("*.csv"))[0]
    print(f"Running MPC on {csv_path.name}...")
    print(f"  K={MPC_K}  H={MPC_H}  σ={MPC_SIGMA}")
    t0 = time.time()
    cost, actions = run_single_route(csv_path)
    dt = time.time() - t0
    print(f"  cost={cost:.1f}  ⏱{dt:.0f}s")
