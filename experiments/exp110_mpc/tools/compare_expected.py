"""Compare temp=0.1 sampled lataccel vs temp=0.8 expected lataccel.
If they're close, feeding E[lataccel] to the policy gives it the "low-temp view"."""

import os
import numpy as np
import sys
from pathlib import Path
from hashlib import md5

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from tinyphysics import (
    TinyPhysicsModel,
    CONTEXT_LENGTH,
    CONTROL_START_IDX,
    COST_END_IDX,
    LATACCEL_RANGE,
    VOCAB_SIZE,
    MAX_ACC_DELTA,
    STEER_RANGE,
)

BINS = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)


def run_segment(data_path, model_path):
    """Run one segment 3 ways: temp=0.1 sampled, temp=0.8 sampled, temp=0.8 expected."""
    import pandas as pd

    mdl = TinyPhysicsModel(model_path, debug=False)
    df = pd.read_csv(data_path)
    data = pd.DataFrame(
        {
            "roll_lataccel": np.sin(df["roll"].values) * 9.81,
            "v_ego": df["vEgo"].values,
            "a_ego": df["aEgo"].values,
            "target_lataccel": df["targetLateralAcceleration"].values,
            "steer_command": -df["steerCommand"].values,
        }
    )

    seed = int(md5(str(data_path).encode()).hexdigest(), 16) % 10**4
    T = len(data)

    def rollout(temperature, use_expected=False):
        np.random.seed(seed)
        state_history = []
        action_history = data["steer_command"].values[:CONTEXT_LENGTH].tolist()
        current_lataccel_history = (
            data["target_lataccel"].values[:CONTEXT_LENGTH].tolist()
        )
        current_lataccel = current_lataccel_history[-1]
        lataccels = []

        for step_idx in range(CONTEXT_LENGTH, T):
            state = data.iloc[step_idx]
            state_history_window = [
                (
                    data.iloc[i]["roll_lataccel"],
                    data.iloc[i]["v_ego"],
                    data.iloc[i]["a_ego"],
                )
                for i in range(max(0, step_idx - CONTEXT_LENGTH + 1), step_idx + 1)
            ]
            # Pad if needed
            while len(state_history_window) < CONTEXT_LENGTH:
                state_history_window.insert(0, state_history_window[0])

            actions_window = action_history[-CONTEXT_LENGTH:]
            preds_window = current_lataccel_history[-CONTEXT_LENGTH:]

            tokenized = np.digitize(np.clip(preds_window, -5, 5), BINS, right=True)
            raw_states = [list(x) for x in state_history_window]
            states = np.column_stack([actions_window, raw_states])
            input_data = {
                "states": np.expand_dims(states, 0).astype(np.float32),
                "tokens": np.expand_dims(tokenized, 0).astype(np.int64),
            }

            res = mdl.ort_session.run(None, input_data)[0]
            probs = mdl.softmax(res / temperature, axis=-1)[0, -1]

            if use_expected:
                pred_val = np.sum(probs * BINS)
            else:
                sample = np.random.choice(VOCAB_SIZE, p=probs)
                pred_val = BINS[sample]

            pred_val = np.clip(
                pred_val,
                current_lataccel - MAX_ACC_DELTA,
                current_lataccel + MAX_ACC_DELTA,
            )

            if step_idx >= CONTROL_START_IDX:
                current_lataccel = pred_val
            else:
                current_lataccel = data.iloc[step_idx]["target_lataccel"]

            current_lataccel_history.append(current_lataccel)
            action_history.append(
                data["steer_command"].values[step_idx]
                if step_idx < CONTROL_START_IDX
                else 0.0
            )  # zero action for comparison
            lataccels.append(current_lataccel)

        return np.array(lataccels)

    print(f"Running {Path(data_path).name}...")
    la_01 = rollout(temperature=0.1, use_expected=False)
    la_08_sampled = rollout(temperature=0.8, use_expected=False)
    la_08_expected = rollout(temperature=0.8, use_expected=True)

    # Also get temp=0.1 expected for completeness
    la_01_expected = rollout(temperature=0.1, use_expected=True)

    # Compare over the control window
    ctrl_slice = slice(
        CONTROL_START_IDX - CONTEXT_LENGTH, COST_END_IDX - CONTEXT_LENGTH
    )
    target = data["target_lataccel"].values[CONTROL_START_IDX:COST_END_IDX]

    def stats(name, la):
        la_ctrl = la[ctrl_slice]
        mse = np.mean((target - la_ctrl) ** 2)
        return name, la_ctrl, mse

    results = [
        stats("temp=0.1 sampled", la_01),
        stats("temp=0.1 expected", la_01_expected),
        stats("temp=0.8 sampled", la_08_sampled),
        stats("temp=0.8 expected", la_08_expected),
    ]

    print(f"\n  {'Method':<25s}  {'MSE vs target':>13s}  {'mean':>7s}  {'std':>7s}")
    print(f"  {'-' * 55}")
    for name, la_ctrl, mse in results:
        print(
            f"  {name:<25s}  {mse:13.4f}  {np.mean(la_ctrl):7.3f}  {np.std(la_ctrl):7.3f}"
        )

    # Pairwise distances
    print(f"\n  Pairwise MAE (control window):")
    names = [r[0] for r in results]
    vals = [r[1] for r in results]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mae = np.mean(np.abs(vals[i] - vals[j]))
            corr = np.corrcoef(vals[i], vals[j])[0, 1]
            print(
                f"    {names[i]:25s} vs {names[j]:25s}  MAE={mae:.4f}  corr={corr:.4f}"
            )

    return results


if __name__ == "__main__":
    model_path = str(ROOT / "models" / "tinyphysics.onnx")
    data_dir = ROOT / "data"
    csvs = sorted(data_dir.glob("*.csv"))[:3]
    for csv in csvs:
        run_segment(str(csv), model_path)
        print()
